using System;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

class Program
{
    // Входные данные (X)
    private static readonly float[] TrainInput = new float[] { 1.0f, 2.0f, -1.0f, 0.5f };
    // Идеальный таргет (Y), который мы хотим получить на выходе Softmax
    private static readonly float[] TrainTarget = new float[] { 0.1f, 0.7f, 0.1f, 0.1f };

    private static readonly int[] DataShape = new[] { 1, 1, 4 }; // [Batch, SeqLen, dModel]

    static unsafe void Main(string[] args)
    {
        Console.WriteLine("=== Запуск честного теста оптимизации cstorchcl ===");

       // try
        {
            using var computeContext = new ComputeContext(preferAccelerator: true);
            using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);

            var nnClGlob = new NnClGlob(computeContext);
            int dModel = 4;

            using var model = new TestLLMBlock(computeContext, nnClGlob, dModel);
            var optimizer = new SGD(model.GetParameters(), learningRate: 0.5f, computeContext); // Увеличим LR для скорости

            Console.WriteLine("\n--- Запуск обучения на парах X -> Y (5 эпох) ---");

            for (int epoch = 1; epoch <= 5; epoch++)
            {
                model.ZeroGrad();
                Console.WriteLine();
                // Выделяем тензоры для текущей итерации
                using var xTensor = new Tensor(TrainInput, DataShape, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);
                using var yTensor = new Tensor(TrainTarget, DataShape, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);

                // 1. Forward Pass
                Tensor pred = model.Forward(xTensor);

                // 2. Расчет ошибки (MSE Loss)
                Tensor lossTensor = nnClGlob.MseLoss(pred, yTensor);
                float[] lossHost = HostReturner.ToCpu(lossTensor);

                // Выводим ошибку и срез проекции весов
                Console.WriteLine($"\n[Эпоха {epoch}] Loss: {lossHost[0]:F6}");
                Console.WriteLine($"            Проекция: {model.ProjectedHost[0]:F4}, {model.ProjectedHost[1]:F4}, {model.ProjectedHost[2]:F4}");

                // 3. Backward Pass от критерия ошибки (Loss)
                AutoGrad.Backward(lossTensor, computeContext.Context);

                // 4. Шаг оптимизатора
                optimizer.Step();
            }

            Console.WriteLine("\n=== Обучение завершено! Проверьте, как падает Loss ===");
        }
       /* catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[ОШИБКА]: {ex.Message}");
            Console.ResetColor();
        }*/
        //finally
        {
            DeviceQueueRegistry.Clear();
        }

        Console.ReadLine();
    }
}
