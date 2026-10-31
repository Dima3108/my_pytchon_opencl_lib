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

        // Добавьте поле в класс LocalOp:
        private readonly ComputeContext _computeCtx;

        // Обновите конструктор класса LocalOp, чтобы он принимал ComputeContext:
        public LocalOp(Tensor[] inputs, Tensor output, string backwardSource, Dictionary<string, Tensor> tensorArgs, ComputeContext computeCtx)
        {
            Inputs = inputs;
            Output = output;
            _backwardKernelSource = backwardSource;
            _tensorArgs = tensorArgs;
            _computeCtx = computeCtx;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
#if DEBUG
            Console.WriteLine("[locRunner]запуск обратного прохода!");
#endif
            var prg = compiler.JITProgramCache.GetOrCompile(context, queue.Device, _backwardKernelSource);
            using Kernel kernel = prg.CreateKernel("fused_elementwise_backward_kernel");

            var outputGrads = new Dictionary<string, Tensor>();
            int argIndex = 0;

            foreach (var kvp in _tensorArgs)
            {
                string name = kvp.Key;
                Tensor t = kvp.Value;

                t.InitGradient(context);

                // Выделяем временный буфер для дельты текущего слоя
                Tensor outLayerGrad = new Tensor(t.Shape, context, MemoryFlags.ReadWrite, requiresGrad: false);
                outputGrads.Add(name, outLayerGrad);

                // =========================================================================
                // ИСПРАВЛЕНИЕ СЦЕПКИ ГРАДИЕНТОВ:
                // Если этот тензор в executionTensors является выходом (например, "P"), 
                // то его входящим градиентом для JIT-математики должен быть gradOutput,
                // который прислал верхний слой (Softmax), а не его собственный внутренний t.Grad!
                // =========================================================================
                NOpenCL.Buffer incomingGradBuffer = t.Grad.DataBuffer;

                // Проверяем, совпадает ли буфер с буфером репрезентативного выхода операции
                if (t.DataBuffer == Output.DataBuffer)
                {
                    incomingGradBuffer = gradOutput.DataBuffer; // Подменяем градиент на пришедший сверху!
                }
                // =========================================================================

                // Передаем аргументы в Си-ядро OpenCL
                kernel.Arguments[argIndex++].SetValue(t.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(incomingGradBuffer); // Используем правильный входящий поток градиента!
                kernel.Arguments[argIndex++].SetValue(outLayerGrad.DataBuffer);
                kernel.Arguments[argIndex++].SetValue(t.Size);
            }

            // Нахождение максимального размера сетки потоков
            int maxTensorSize = _tensorArgs.Values.Max(t => t.Size);

            int warp = _computeCtx.warp_size;
            // =========================================================================
            // ИСПРАВЛЕНИЕ: Точное выравнивание 2D варп-сетки для обратного прохода
            // =========================================================================
            //int maxTensorSize = _tensorArgs.Values.Max(t => t.Size);

            
            int localX = _computeCtx.MAX_WORK_GPOUP_SIZE_X; // 256
            int localY = _computeCtx.MAX_WORK_GPOUP_SIZE_Y; // 4

            // Вычисляем глобальные размеры строго по вашей формуле
            int rawGlobalY = warp;
            int rawGlobalX = 1 + (maxTensorSize / rawGlobalY);

            // Округляем вверх до кратности локальным группам
            int globalX = ((rawGlobalX + localX - 1) / localX) * localX;
            int globalY = ((rawGlobalY + localY - 1) / localY) * localY;

            IntPtr[] globalWorkSize = new[] { new IntPtr(globalX), new IntPtr(globalY) };
            IntPtr[] localWorkSize = new[] { new IntPtr(localX), new IntPtr(localY) }; // Передаем X и Y без путаницы!
            // =========================================================================

            queue.EnqueueNDRangeKernel(kernel, null, globalWorkSize, localWorkSize);
            queue.Finish();

            queue.Finish();

            // Формируем результирующий массив градиентов для возврата в AutoGrad
            // ... (после выполнения ядра queue.Finish(); на странице 3)

            // Формируем чистый результирующий массив изолированных градиентов для возврата в AutoGrad
            Tensor[] finalResultGrads = new Tensor[Inputs.Length];

            for (int i = 0; i < Inputs.Length; i++)
            {
                Tensor inputTensor = Inputs[i];
                Tensor calculatedGradForLayer = null;

                // Находим, под каким строковым именем этот входной тензор шел в JIT-блоке
                foreach (var kvp in _tensorArgs)
                {
                    if (kvp.Value == inputTensor)
                    {
                        // Извлекаем буфер градиента, который посчитало наше Си-ядро
                        calculatedGradForLayer = outputGrads[kvp.Key];
                        break;
                    }
                }

                if (calculatedGradForLayer != null)
                {
                    // Важный шаг (как у BmmOp): сбрасываем создателя, чтобы этот тензор градиента 
                    // считался чистыми данными и не участвовал в обходе графа
                    calculatedGradForLayer.Creator = null;
                    finalResultGrads[i] = calculatedGradForLayer;
                }
                else
                {
                    finalResultGrads[i] = null;
                }
            }

            // Возвращаем изолированные градиенты. Движок AutoGrad сам сложит их!
            return finalResultGrads;
        }

    }

}

