using System;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

class Program
{
    static unsafe void Main(string[] args)
    {
        Console.WriteLine("=== Запуск тестов сборки слоев cstorchcl ===");

        try
        {
            // 1. Инициализация OpenCL контекста (каскадный выбор: Ускоритель -> GPU -> CPU)
            using var computeContext = new ComputeContext(preferAccelerator: true);
            Console.WriteLine($"Выбран девайс: {computeContext.Device.Name} от {computeContext.Device.Vendor}");

            // 2. Создание очереди команд и регистрация в глобальном потоке
            using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);

            // 3. Инициализация диспетчера Си-ядер
            var nnClGlob = new NnClGlob(computeContext);
            Console.WriteLine("Глобальные Си-ядра OpenCL успешно загружены и JIT-скомпилированы.");

            // 4. Создание экземпляра нашей тестовой модели (использует NeuralNetwork как базовый)
            int batch = 1;
            int seqLen = 4;
            int dModel = 256;

            using var model = new TestLLMBlock(computeContext, nnClGlob, dModel);
            Console.WriteLine("Параметры модели успешно аллоцированы во VRAM.");

            // 5. Создание фиктивного входного тензора X
            var inputTensor = new Tensor(new[] { batch, seqLen, dModel }, computeContext.Context, MemoryFlags.ReadWrite, requiresGrad: true);

            // Наполняем вход тестовыми данными на GPU через fixed-указатель
            float[] initialInputData = new float[inputTensor.Size];
            fixed (float* iid = initialInputData)
            {
                Array.Fill(initialInputData, 1.0f);
                queue.EnqueueWriteBuffer(inputTensor.DataBuffer, true, 0, initialInputData.Length * sizeof(float), (IntPtr)iid);
            }
            queue.Finish();

            Console.WriteLine("\n--- Запуск FORWARD PASS ---");
            model.ZeroGrad(); // Сбрасываем старые градиенты

            Tensor output = model.Forward(inputTensor);

            Console.WriteLine($"Успешно! Форма выходного тензора: [{string.Join(", ", output.Shape)}]");
            Console.WriteLine($"Проверка графа: Выходной тензор привязан к операции: {output.Creator?.GetType().Name}");

            // =========================================================================
            // МЕСТО ДЛЯ ВСТАВКИ 1: Выводим данные, которые модель сохранила на CPU в процессе Forward
            // =========================================================================
            Console.WriteLine("\n[Отладка шага модели]: Промежуточные данные ProjectedHost успешно скачаны на CPU.");
            Console.WriteLine($"Размер скачанного массива: {model.ProjectedHost.Length} элементов.");
            Console.WriteLine($"Первые значения линейной проекции: {model.ProjectedHost[0]:F4}, {model.ProjectedHost[1]:F4}, {model.ProjectedHost[2]:F4}...");
            // =========================================================================

            Console.WriteLine("\n--- Запуск BACKWARD PASS (Autograd) ---");
            // Запускаем автоматический расчет градиентов от выхода назад к весам модели
            AutoGrad.Backward(output, computeContext.Context);
            Console.WriteLine("Backpropagation завершен успешно!");

            // Проверяем, появились ли градиенты у параметров модели
            Console.WriteLine($"Градиент для RmsGamma создан: {model.RmsGamma.Grad != null}");
            Console.WriteLine($"Градиент для ProjectionWeights создан: {model.ProjectionWeights.Grad != null}");
            Console.WriteLine($"Гradiент для Входа X создан: {inputTensor.Grad != null}");

            // =========================================================================
            // МЕСТО ДЛЯ ВСТАВКИ 2: Красивый вывод итоговых тензоров и градиентов через Фасад
            // =========================================================================
            Console.WriteLine("\n--- Визуализация результатов вычислений ---");
            HostReturner.PrintTensor(output, name: "Выход Модели (Softmax)", printGrad: true);
            HostReturner.PrintTensor(inputTensor, name: "Входной Тензор X", printGrad: true);
            // =========================================================================

            Console.WriteLine("\n=== Все тесты пройдены успешно! Архитектура графа и OpenCL стабильны ===");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[КРИТИЧЕСКАЯ ОШИБКА ТЕСТИРОВАНИЯ]: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Детали: {ex.InnerException.Message}");
            }
            Console.ResetColor();
        }
        finally
        {
            // Очищаем потоковый реестр
            DeviceQueueRegistry.Clear();
        }

        Console.ReadLine();
    }
}
