using System;
using System.Collections.Generic;
using NOpenCL;

namespace cstorchcl.core
{
    /// <summary>
    /// Динамический узел графа автоматического дифференцирования для JIT-сплавленных ядер LocalRunner.
    /// </summary>
    public class LocalOp : Op
    {
        private readonly string _backwardKernelSource;
        private readonly Dictionary<string, Tensor> _tensorArgs;

        public LocalOp(Tensor[] inputs, Tensor output, string backwardSource, Dictionary<string, Tensor> tensorArgs)
        {
            Inputs = inputs;
            Output = output;
            _backwardKernelSource = backwardSource;
            _tensorArgs = tensorArgs;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Компилируем или извлекаем из кэша наше кастомное обратное JIT-ядро
            var prg = compiler.JITProgramCache.GetOrCompile(context, queue.Device, _backwardKernelSource);
            using Kernel kernel = prg.CreateKernel("fused_elementwise_backward_kernel");

            // Создаем выходные тензоры градиентов для каждого задействованного входа
            var outputGrads = new Dictionary<string, Tensor>();
            // Внутри метода LocalOp.Backward:
            int argIndex = 0;
            foreach (var kvp in _tensorArgs)
            {
                string name = kvp.Key;
                Tensor t = kvp.Value;

                // 1. Инициализируем градиентный буфер параметра, если он пустой
                t.InitGradient(context);

                // 2. Создаем временный выходной буфер текущего слоя для аккумуляции
                Tensor outLayerGrad = new Tensor(t.Shape, context, MemoryFlags.ReadWrite, requiresGrad: false);
                outputGrads.Add(name, outLayerGrad);

                // ПРИВЯЗКА СТРОГО ПО НАШЕЙ ИСПРАВЛЕННОЙ СИГНАТУРЕ:
                // global float* {name}, global float* grad_{name}, global float* out_grad_{name}, const int {name}_len
                kernel.Arguments[argIndex++].SetValue(t.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(t.Grad.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(outLayerGrad.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(t.Size);
            }


            // Находим максимальный размер сетки потоков
            int maxTensorSize = 0;
            foreach (var t in _tensorArgs.Values)
                if (t.Size > maxTensorSize) maxTensorSize = t.Size;

            IntPtr globalSize = new IntPtr(maxTensorSize);
            queue.EnqueueNDRangeKernel(kernel,  new[] { globalSize }, null);
            queue.Finish();

            // Формируем результирующий массив градиентов, сопоставляя их с базовым порядком в Inputs
            Tensor[] finalResultGrads = new Tensor[Inputs.Length];
            for (int i = 0; i < Inputs.Length; i++)
            {
                // Находим имя тензора в словаре аргументов
                foreach (var kvp in _tensorArgs)
                {
                    if (kvp.Value == Inputs[i])
                    {
                        finalResultGrads[i] = outputGrads[kvp.Key];
                        break;
                    }
                }
            }

            return finalResultGrads;
        }
    }
}
