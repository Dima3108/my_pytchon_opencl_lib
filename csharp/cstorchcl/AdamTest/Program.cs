using System;
using System.Collections.Generic;
using cstorchcl.core;
using cstorchcl.optimizers;
using cstorchcl.compiler;
namespace TrainTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== СТАРТ ТЕСТИРОВАНИЯ ОПТИМИЗАТОРА ADAM ===");

            // 1. Инициализируем контекст OpenCL (preferAccelerator: true для запуска на GPU)
            using var computeCtx = new ComputeContext(preferAccelerator: true);

            // Регистрируем очередь выполнения в потоке (чтобы работал DeviceQueueRegistry)
            // Примечание: Убедись, что при создании контекста ты вызываешь метод регистрации, 
            // например: DeviceQueueRegistry.RegisterCurrent(computeCtx.CreateQueue());
            // Для теста предполагаем, что очередь уже активна внутри ComputeContext.
            var context = computeCtx.Context;
            DeviceQueueRegistry.RegisterCurrent(context.CreateCommandQueue(computeCtx.Device));
            // 2. Создаем обучаемые параметры (например, веса модели)
            // Форма: [1, 2, 4] - 3D тензор, как ожидает твой Bmm (страница 24)
            int[] shape = { 1, 2, 4 };

            // Обучаемый тензор (наши веса, которые будем подгонять)
            Tensor weights = new Tensor(shape, context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad: true);
            // Целевой тензор (то, к чему должны прийти веса)
            Tensor target = new Tensor(shape, context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad: false);
            // Входные данные X
            Tensor inputX = new Tensor(shape, context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad: false);

            // 3. Заполняем тестовыми данными (через промежуточные массивы на CPU)
            float[] initWeights = { 0.5f, -0.2f, 0.1f, 0.9f, 0.3f, 0.7f, -0.5f, 0.2f };
            float[] targetWeights = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f };
            float[] inputsData = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

            var queue = DeviceQueueRegistry.GetCurrent();
            unsafe
            {
                fixed (float* pW = initWeights, pT = targetWeights, pX = inputsData)
                {
                    queue.EnqueueWriteBuffer(weights.DataBuffer, true, 0, initWeights.Length * sizeof(float), (IntPtr)pW);
                    queue.EnqueueWriteBuffer(target.DataBuffer, true, 0, targetWeights.Length * sizeof(float), (IntPtr)pT);
                    queue.EnqueueWriteBuffer(inputX.DataBuffer, true, 0, inputsData.Length * sizeof(float), (IntPtr)pX);
                }
            }
            queue.Finish();

            // 4. Создаем оптимизатор Adam и передаем параметры на обучение
            List<Tensor> paramsToOptimize = new List<Tensor> { weights };
            // Задаем lr повыше (например, 0.05), чтобы на синтетическом тесте лосс упал за пару десятков эпох
             Adam optimizer = new Adam(paramsToOptimize, lr: 0.05f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f);
           // var optimizer = new SGD(paramsToOptimize, 0.05f, computeCtx);
            Console.WriteLine("Начальные веса успешно загружены во VRAM.");
            Console.WriteLine("Запуск цикла оптимизации (20 эпох)...");
            Console.WriteLine("--------------------------------------------");

            // Инициализируем глобальный диспетчер операций
            NnClGlob clGlob = new NnClGlob(computeCtx);

            // 5. Обучающий цикл
            for (int epoch = 1; epoch <= 20; epoch++)
            {
                // Обнуляем градиенты на каждом шаге
                optimizer.ZeroGrad();
               
                // Forward Pass: Считаем ошибку MSE между текущими весами и целевыми.
                // Примечание: Для чистоты проверки Adam мы можем считать ошибку напрямую 
                // между тензором весов и таргетом.
                Tensor loss = clGlob.MseLoss(weights, target);

                // Скачиваем значение лосса на CPU для логирования
                // Используем твой HostReturner или EnqueueReadBuffer напрямую:
                float[] lossHost = new float[1];
                unsafe
                {
                    fixed (float* pL = lossHost)
                    {
                        queue.EnqueueReadBuffer(loss.DataBuffer, true, 0, sizeof(float), (IntPtr)pL);
                    }
                }
                queue.Finish();

                Console.WriteLine($"Эпоха {epoch:D2} | Текущий MSE Loss: {lossHost[0]:F6}");

                // Backward Pass: Строим граф автодифференцирования от точки Loss
                AutoGrad.Backward(loss, context);

                // Optimization Step: Обновление весов с помощью нашего кастомного Adam ядра
                optimizer.Step();
            }

            Console.WriteLine("--------------------------------------------");
            Console.WriteLine("Цикл завершен. Проверяем финальные веса на GPU...");

            // 6. Выкачиваем обученные веса обратно на CPU, чтобы убедиться, что они сдвинулись к таргету
            float[] finalWeights = new float[weights.Size];
            unsafe
            {
                fixed (float* pW = finalWeights)
                {
                    queue.EnqueueReadBuffer(weights.DataBuffer, true, 0, finalWeights.Length * sizeof(float), (IntPtr)pW);
                }
            }
            queue.Finish();

            Console.WriteLine("\nСравнение результатов (Target vs Trained):");
            for (int i = 0; i < finalWeights.Length; i++)
            {
                Console.WriteLine($"Элемент [{i}]: Идеал = {targetWeights[i]:F2} | Обучено = {finalWeights[i]:F4}");
            }

            Console.WriteLine("\n=== ТЕСТ ЗАВЕРШЕН ===");
        }
    }
}
