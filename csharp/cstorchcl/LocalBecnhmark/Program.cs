using System;
using System.Diagnostics;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== ЗАПУСК БЕНЧМАРКА: СЛИЯНИЕ ЯДЕР GPU (JIT FUSION) ===");

        //try
        {
            // 1. Инициализация OpenCL контекста и очереди
            using var computeContext = new ComputeContext(preferAccelerator: true);
            using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);

            Console.WriteLine($"Устройство: {computeContext.Device.Name}");

            // Размер тензоров возьмем побольше, чтобы GPU успевал разгоняться, 
            // но при этом сохраним разные длины векторов для проверки стабильности конвейера
            int sizeA = 100_000;
            int sizeB = 80_000;
            int sizeC = 120_000;

            // Выделяем тензоры в режиме EVAL (requiresGrad = false)
            float[] initDataA = new float[sizeA]; Array.Fill(initDataA, 1.0f);
            float[] initDataB = new float[sizeB]; Array.Fill(initDataB, 0.5f);
            float[] initDataC = new float[sizeC]; Array.Fill(initDataC, -1.0f);

            using var tensorA = new Tensor(initDataA, new[] { sizeA }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            using var tensorB = new Tensor(initDataB, new[] { sizeB }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            using var tensorC = new Tensor(initDataC, new[] { sizeC }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            // Создаем символические обертки для LocalRunner
            var A = new LocalVariable(tensorA, "A");
            var B = new LocalVariable(tensorB, "B");
            var C = new LocalVariable(tensorC, "C");

            float q = 1.5f;
            uint seed = 42;
            int iterations = 1000;

            // 1. Создаем словарь связей строковых имен и физических тензоров GPU
            var tensorArgs = new Dictionary<string, Tensor> {
                { "A", tensorA },
                { "B", tensorB },
                { "C", tensorC }
            };

            // 2. Создаем словарь захваченных внешних констант C#
            var constants = new Dictionary<string, float> {
                { "q", 1.5f }
            };

            // 3. Описываем монолитную пачку математики в виде простого текстового блока (Псевдокод)
            string codeBlock = @"
                A = nnclloc.tanh(A);
                C = nnclloc.cos(B);
                A = A / q;
                C = nnclloc.dropout(C, 0.2, 42);
                B = B + 0.01;
            ";

            Console.WriteLine("Тест 2: Запуск 1000 итераций через промышленный Roslyn JIT-Fusion...");

            Stopwatch swFused = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                // Запуск конвейера. Roslyn мгновенно соберет код ядра, 
                // а JITProgramCache выдаст уже скомпилированный бинарник со 2-й итерации!
                NnClLoc.Run(codeBlock, tensorArgs, constants);
            }
            queue.Finish();
            swFused.Stop();



            // =========================================================================
            // ТЕСТ 2: ОДИН МОНОЛИТНЫЙ КОНВЕЙЕР В ЦИКЛЕ (С ПОЛНЫМ JIT-СЛИЯНИЕМ)
            // =========================================================================





            // =========================================================================
            // СБОР СТАТИСТИКИ И ВЫВОД РЕЗУЛЬТАТОВ
            // =========================================================================
           // double totalTimeUnfusedMs = swUnfused.Elapsed.TotalMilliseconds;
            //double avgTimeUnfusedMs = totalTimeUnfusedMs / iterations;

            double totalTimeFusedMs = swFused.Elapsed.TotalMilliseconds;
            double avgTimeFusedMs = totalTimeFusedMs / iterations;

            //double speedup = totalTimeUnfusedMs / totalTimeFusedMs;

            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ");
            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"Раздельные ядра (Unfused):");
            //Console.WriteLine($"  -> Общее время:  {totalTimeUnfusedMs:F2} мс");
           // Console.WriteLine($"  -> Среднее на итерацию: {avgTimeUnfusedMs:F4} мс");

            Console.WriteLine($"\nСплавленное ядро (Fused JIT):");
            Console.WriteLine($"  -> Общее время:  {totalTimeFusedMs:F2} мс");
            Console.WriteLine($"  -> Среднее на итерацию: {avgTimeFusedMs:F4} мс");

            Console.WriteLine(new string('-', 60));
            Console.ForegroundColor = ConsoleColor.Green;
            //Console.WriteLine($"Ускорение за счет Fusion: {speedup:F2}x раз(а)!");
            Console.ResetColor();
            Console.WriteLine(new string('=', 60));

        }
        //catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
          //  Console.WriteLine($"\n[КРИТИЧЕСКАЯ ОШИБКА БЕНЧМАРКА]: {ex.Message}");
            Console.ResetColor();
        }

        Console.ReadLine();
    }
}
