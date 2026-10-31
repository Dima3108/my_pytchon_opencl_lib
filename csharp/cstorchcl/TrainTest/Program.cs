using System;
using System.Collections.Generic;
using System.Linq;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== ТЕСТИРОВАНИЕ ОБУЧАЕМОГО СЛОЯ LINEAR (OPENCL JIT) ===");

        try
        {
            // 1. Инициализация OpenCL
            using var computeContext = new ComputeContext(preferAccelerator: true);
            using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);

            Console.WriteLine($"Выбран девайс: {computeContext.Device.Name}");

            // Геометрия: Батч = 2 примера, Входных фичей = 3, Выходных фичей = 4
            int batch = 2; int inFeatures = 3; int outFeatures = 4;

            float[] inputData = new float[] {
                1.0f, 2.0f, 3.0f,  // Пример 1
                4.0f, 5.0f, 6.0f   // Пример 2
            };
            float[] targetData = new float[] {
                0.0f, 1.0f, 0.0f, 0.0f,  // Желаемый выход для Примера 1
                1.0f, 0.0f, 0.0f, 0.0f   // Желаемый выход для Примера 2
            };

            // Инициализация тензоров данных
            using var xTensor = new Tensor(inputData, new[] { batch, 1, inFeatures }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);
            using var targetTensor = new Tensor(targetData, new[] { batch, 1, outFeatures }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            var nnClGlob = new NnClGlob(computeContext);

            // 2. Создание нашего нового слоя Linear!
            using var linearLayer = new Linear(computeContext, nnClGlob, inFeatures, outFeatures);

            // Передаем параметры слоя в оптимизатор SGD
            var optimizer = new SGD(linearLayer.GetParameters(), learningRate: 0.2f, computeContext);

            Console.WriteLine("\n--- Выполнение шага обучения слоя Linear ---");
            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine($"эпоха {epoch+1}!");
                GradientStorage.Clear();
                linearLayer.ZeroGrad(); // Сбрасываем старые буферы градиентов параметров

                // А. Прямой проход слоя (BMM + JIT Bias)
                Tensor pred = linearLayer.Forward(xTensor);
                Console.WriteLine($"[Forward] Выходной тензор успешно рассчитан. Форма: [{string.Join(", ", pred.Shape)}]");

                // Б. Расчет величины ошибки
                Tensor loss = nnClGlob.MseLoss(pred, targetTensor);
                float[] lossCpu = HostReturner.ToCpu(loss);
                Console.WriteLine($"[Loss] Текущая ошибка слоя: {lossCpu[0]:F6}");

                // В. ЗАПУСК ОБРАТНОГО ПРОХОДА (Autograd)
                // Градиент пойдет: MSE -> LocalOp (дельта по bias) -> BmmOp -> Веса Weight -> Входной тензор X
                AutoGrad.Backward(loss, computeContext.Context);
                Console.WriteLine("[Backward] Каскадный обратный проход графа успешно завершен!");

                // Г. Проверка сцепки буферов градиентов на GPU во VRAM
                Console.WriteLine("\nАнализ буферов градиентов после сцепки:");
                Console.WriteLine($"  -> Градиент для смещения Bias создан: {linearLayer.Bias.Grad != null}");
                Console.WriteLine($"  -> Градиент для матрицы весов Weight создан: {linearLayer.Weight.Grad != null}");
                Console.WriteLine($"  -> Градиент дотек до самого первого входа X: {xTensor.Grad != null}");

                if (linearLayer.Bias.Grad != null && linearLayer.Weight.Grad != null)
                {
                    float[] biasGrad = HostReturner.ToCpuGrad(linearLayer.Bias);
                    float[] weightGrad = HostReturner.ToCpuGrad(linearLayer.Weight);

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"\n[Успех] Градиент Bias (все 4 эл.): [{string.Join(", ", biasGrad.Select(x => x.ToString("F4")))}]");
                    Console.WriteLine($"        Градиент весов Weight (первые 4): {weightGrad[0]:F5}, {weightGrad[1]:F5}, {weightGrad[2]:F5}, {weightGrad[3]:F5}");
                    Console.ResetColor();

                    // Д. Физическое обновление параметров во VRAM
                    optimizer.Step();
                    Console.WriteLine("\n[SGD] Шаг оптимизации успешно применен. Веса слоя Linear обновлены на GPU.");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("\n[Ошибка] Градиенты застряли на стыке слоев!");
                    Console.ResetColor();
                }
}
                Console.WriteLine("\n=== ТЕСТ ОБУЧАЕМОГО СЛОЯ LINEAR С УСПЕХОМ ЗАВЕРШЕН! ===");
            
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[КРИТИЧЕСКАЯ ОШИБКА]: {ex.Message}");
            Console.ResetColor();
        }
        finally
        {
            DeviceQueueRegistry.Clear();
        }

        Console.ReadLine();
    }
}
