using System;
using System.Collections.Generic;
using System.Linq;
using NOpenCL;
using cstorchcl.core;

namespace cstorchcl.compiler
{
    public static class NnClLoc
    {
        /// <summary>
        /// Выполняет динамическую сборку и запуск сплавленного OpenCL-ядра на 2D-сетке варпов.
        /// </summary>
        ///         /// <summary>
        /// Выполняет динамическую сборку и возвращает СЛОВАРЬ новых выходных тензоров.
        /// </summary>
        public static Dictionary<string, Tensor> Run(string pseudoCode, Dictionary<string, Tensor> tensors, ComputeContext computeCtx, Dictionary<string, float> constants = null)
        {
            if (string.IsNullOrWhiteSpace(pseudoCode)) return null;
            if (tensors == null || tensors.Count == 0) return null;

            var transpiler = new LocalRoslynTranspiler();
            var (forwardSource, backwardSource) = transpiler.TranslateWithGrad(pseudoCode, constants, tensors);

            int N = tensors.Values.Max(t => t.Size);
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            Context context = queue.Context;

            // =========================================================================
            // ИСПРАВЛЕНИЕ: Выделяем новые изолированные буферы для результатов вычислений
            // =========================================================================
            var outputDict = new Dictionary<string, Tensor>(StringComparer.OrdinalIgnoreCase);

            foreach (var outName in transpiler.OutputTensorNames)
            {
                Tensor originalIn = tensors[outName];

                // Создаем НОВЫЙ тензор. Он будет чистым результатом JIT-блока!
                Tensor newOut = new Tensor(originalIn.Shape, context, MemoryFlags.ReadWrite,
                    requiresGrad: tensors.Values.Any(t => t.RequiresGrad));

                outputDict.Add(outName, newOut);
            }

            // Создаем рабочий словарь для запуска ядра на GPU.
            // Заменяем выходные буферы на новые чистые структуры, чтобы не затирать входы!
            var executionTensors = new Dictionary<string, Tensor>(tensors, StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in outputDict)
            {
                executionTensors[kvp.Key] = kvp.Value;
            }
            // =========================================================================

            int warp = computeCtx.warp_size;
            int globalX = ((1 + (N / warp)) + computeCtx.MAX_WORK_GPOUP_SIZE_X - 1) / computeCtx.MAX_WORK_GPOUP_SIZE_X * computeCtx.MAX_WORK_GPOUP_SIZE_X;
            int globalY = ((warp) + computeCtx.MAX_WORK_GPOUP_SIZE_Y - 1) / computeCtx.MAX_WORK_GPOUP_SIZE_Y * computeCtx.MAX_WORK_GPOUP_SIZE_Y;

            Program program = JITProgramCache.GetOrCompile(context, queue.Device, forwardSource);
            using Kernel kernel = program.CreateKernel("fused_elementwise_kernel");

            int argIndex = 0;
            foreach (var tensorName in transpiler.DiscoveredTensors)
            {
                Tensor t = executionTensors[tensorName]; // Читает из старых, пишет в новые!
                kernel.Arguments[argIndex++].SetValue(t.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(t.Size);
            }

            queue.EnqueueNDRangeKernel(kernel, null, new[] { new IntPtr(globalX), new IntPtr(globalY) }, new[] { new IntPtr(computeCtx.MAX_WORK_GPOUP_SIZE_X), new IntPtr(computeCtx.MAX_WORK_GPOUP_SIZE_Y) });
            queue.Finish();

            // Интеграция с Autograd
            bool anyRequiresGrad = tensors.Values.Any(t => t.RequiresGrad);
            // ИСПРАВЛЕНИЕ ИНТЕГРАЦИИ: Убираем дублирование слоев в Autograd
         
            if (anyRequiresGrad)
            {
                Tensor[] inputs = tensors.Values.ToArray();
                Tensor representative = outputDict.Values.First();

                var localOp = new LocalOp(inputs, representative, backwardSource, executionTensors, computeCtx);

                // Назначаем Creator СТРОГО только выходным тензорам JIT-блока!
                // Входной projected сохранит свой BmmOp, а Bias останется листом графа (Creator = null)
                foreach (var outTensor in outputDict.Values)
                {
                    outTensor.Creator = localOp;
                }
            }


            return outputDict; // Возвращаем коллекцию новых результатов
        }


    }
}
