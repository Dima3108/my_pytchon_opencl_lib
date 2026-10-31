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
        /// Выполняет динамическую сборку и запуск сплавленного OpenCL-ядра (Forward + Backward).
        /// </summary>
        public static void Run(string pseudoCode, Dictionary<string, Tensor> tensors, Dictionary<string, float> constants = null)
        {
            if (string.IsNullOrWhiteSpace(pseudoCode)) return;
            if (tensors == null || tensors.Count == 0) return;

            // 1. Запуск транслятора Roslyn (Генерируем сразу Forward и Backward исходники)
            var transpiler = new LocalRoslynTranspiler();
            var (forwardSource, backwardSource) = transpiler.TranslateWithGrad(pseudoCode, constants);

            // 2. Вычисляем максимальный размер сетки потоков GPU
            int maxTensorSize = 0;
            foreach (var tensorName in transpiler.DiscoveredTensors)
            {
                if (tensors.TryGetValue(tensorName, out var t))
                {
                    if (t.Size > maxTensorSize) maxTensorSize = t.Size;
                }
                else
                {
                    throw new InvalidOperationException($"Критическая ошибка JIT: Тензор '{tensorName}' не передан в аргументы.");
                }
            }

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            Context context = queue.Context;
            Device device = queue.Device;

            // 3. Компиляция и запуск прямого прохода (Forward)
            Program program = JITProgramCache.GetOrCompile(context, device, forwardSource);
            using Kernel kernel = program.CreateKernel("fused_elementwise_kernel");

            int argIndex = 0;
            foreach (var tensorName in transpiler.DiscoveredTensors)
            {
                Tensor t = tensors[tensorName];
                kernel.Arguments[argIndex++].SetValue(t.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(t.Size);
            }

            IntPtr globalSize = new IntPtr(maxTensorSize);
            queue.EnqueueNDRangeKernel(kernel, new[] { globalSize }, null);
            queue.Finish();

            // 4. ИНТЕГРАЦИЯ С AUTOGRAD: Если хотя бы один тензор обучаемый — встраиваем LocalOp в граф
            bool anyRequiresGrad = tensors.Values.Any(t => t.RequiresGrad);
            if (anyRequiresGrad)
            {
                Tensor[] inputs = tensors.Values.ToArray();

                // Создаем один общий узел операции для всей группы сплавленных вычислений.
                // В качестве репрезентативного выхода передаем тензор, который является целью (например, первый измененный)
                Tensor representativeOutput = tensors.Values.First();
                var localOp = new LocalOp(inputs, representativeOutput, backwardSource, tensors);

                // ВАЖНОЕ ИСПРАВЛЕНИЕ: Мы должны назначить этот Creator ВСЕМ тензорам, 
                // которые участвовали в данном JIT-конвейере и требуют расчета градиентов,
                // чтобы AutoGrad гарантированно споткнулся об эту операцию при движении назад от Loss!
                foreach (var kvp in tensors)
                {
                    if (kvp.Value.RequiresGrad)
                    {
                        kvp.Value.Creator = localOp;
                    }
                }
            }

        }
    }
}
