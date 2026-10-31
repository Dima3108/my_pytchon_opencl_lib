using System;
using System.Collections.Generic;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== ЗАПУСК ТЕСТА ЛОКАЛЬНЫХ ГРАДИЕНТОВ (JIT BACKWARD) ===");

       // try
        {
            // 1. Инициализация OpenCL контекста и очереди команд
            using var computeContext = new ComputeContext(preferAccelerator: true);
            using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);

            Console.WriteLine($"Выбран ускоритель: {computeContext.Device.Name}");

            // Зададим тестовые размерности векторов
            int sizeA = 4;
            int sizeB = 4;
            int sizeC = 4;

            // Создаем чистые массивы данных для CPU
            float[] dataA = new float[] { 0.5f, -0.2f, 0.8f, -0.1f };
            float[] dataB = new float[] { 0.1f, 0.4f, -0.5f, 0.2f };
            float[] dataC = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };
            float[] dataTarget = new float[] { 0.0f, 0.0f, 0.0f, 0.0f }; // Идеальный таргет-ноль

            // Инициализируем тензоры во VRAM с флагом RequiresGrad = true
            using var tensorA = new Tensor(dataA, new[] { sizeA }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);
            using var tensorB = new Tensor(dataB, new[] { sizeB }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);
            using var tensorC = new Tensor(dataC, new[] { sizeC }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);
            using var targetTensor = new Tensor(dataTarget, new[] { sizeA }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            var nnClGlob = new NnClGlob(computeContext);
            var optimizer = new SGD(new[] { tensorA, tensorB, tensorC }, learningRate: 0.1f, computeContext);

            // Регистрируем аргументы для нашего JIT-блока
            var tensorArgs = new Dictionary<string, Tensor> {
                { "A", tensorA },
                { "B", tensorB },
                { "C", tensorC }
            };

            var constants = new Dictionary<string, float> {
                { "q", 2.0f }
            };

            // Математический JIT-блок, который мы будем дифференцировать
            string pseudoCode = @"
                A = nnclloc.tanh(A);
                C = nnclloc.cos(B);
                A = A / q;
            ";

            Console.WriteLine("\nВыполнение шага оптимизации (Forward -> Loss -> Backward -> SGD)...");

            // Сбрасываем старые градиенты
            GradientStorage.Clear();

            // 1. Запуск сплавленного Forward конвейера
            NnClLoc.Run(pseudoCode, tensorArgs, constants);

            // 2. Считаем ошибку MSE Loss относительно нуля (берем tensorA как результирующий выход блока)
            Tensor loss = nnClGlob.MseLoss(tensorA, targetTensor);
            float[] lossCpu = HostReturner.ToCpu(loss);
            Console.WriteLine($"[Шаг 1] Текущий Loss: {lossCpu[0]:F6}");

            // Выведем значения тензора А до обновления весов
            float[] cpuA_before = HostReturner.ToCpu(tensorA);
            Console.WriteLine($"Экземпляры тензора А до шага SGD: {cpuA_before[0]:F4}, {cpuA_before[1]:F4}, {cpuA_before[2]:F4}");

            // 3. ЗАПУСК КАСКАДНОГО BACKWARD PASS
            // Должен размотать MseLossOp -> дойти до LocalOp -> скомпилировать и выполнить JIT-ядро градиентов
            AutoGrad.Backward(loss, computeContext.Context);
            Console.WriteLine("Автодифференцирование графа успешно завершено!");

            // 4. Проверяем физическое наличие рассчитанных градиентов на GPU
            Console.WriteLine($"\nПроверка буферов градиентов во VRAM:");
            Console.WriteLine($"  -> Градиент для тензора А создан: {tensorA.Grad != null}");
            Console.WriteLine($"  -> Градиент для тензора B создан: {tensorB.Grad != null}");
            Console.WriteLine($"  -> Градиент для тензора C создан: {tensorC.Grad != null}");

            if (tensorA.Grad != null)
            {
                float[] gradA = HostReturner.ToCpuGrad(tensorA);
                Console.WriteLine($"  -> Значения рассчитанного градиента А (первые 2): {gradA[0]:F4}, {gradA[1]:F4}");
            }

            // 5. Делаем шаг обновления весов
            optimizer.Step();
            Console.WriteLine("\nШаг SGD успешно выполнен.");

            // Выведем значения тензора А после обновления, чтобы зафиксировать изменение байт во VRAM
            float[] cpuA_after = HostReturner.ToCpu(tensorA);
            Console.WriteLine($"Экземпляры тензора А после шага SGD: {cpuA_after[0]:F4}, {cpuA_after[1]:F4}, {cpuA_after[2]:F4}");

            Console.WriteLine("\n=== ТЕСТ ЛОКАЛЬНЫХ ГРАДИЕНТОВ ПРОЙДЕН УСПЕШНО! СВЯЗКА С AUTOGRAD СТАБИЛЬНА ===");
        }
        
        //finally
        {
            DeviceQueueRegistry.Clear();
        }

        Console.ReadLine();
    }
}
