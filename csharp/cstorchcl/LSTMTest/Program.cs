using cstorchcl.compiler;
using cstorchcl.core;
using cstorchcl.layers;
using cstorchcl.optimizers;
using cstorchcl.tests.cstorchcl.tests;
using NOpenCL;
using System;
namespace cstorchcl.tests
{
    public static class LstmDebugger
    {
        public static void RunLstmTest()
        {
            Console.WriteLine("=== ЗАПУСК ТЕСТА МОНОЛИТНОГО СЛОЯ LSTM ===");
            ComputeContext computeContext = new ComputeContext();
          
            DeviceQueueRegistry.RegisterCurrent(computeContext.Context.CreateCommandQueue(computeContext.Device));
  NnClGlob nnClGlob = new NnClGlob(computeContext);
            // 1. Инициализация базового OpenCL контекста через ваши синглтоны/реестры
            // (Подставьте ваши актуальные объекты получения контекста, например DeviceQueueRegistry)
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            Context openClContext = activeQueue.Context;

            // Предполагаем, что экземпляр диспетчера ядер NnClGlob уже инициализирован
            // Нам понадобятся ComputeContext и диспетчер
            // var computeCtx = ...
            // var nnClGlob = ...

            // 2. Конфигурация тестовых размерностей
            int batchSize = 2;   // Ботчи (Batch)
            int seqLen = 3;      // Длина последовательности (T)
            int inFeatures = 4;  // Размерность входного вектора (X_dim)
            int hiddenSize = 3;  // Скрытая размерность LSTM (H_dim)

            Console.WriteLine($"Конфигурация: Batch={batchSize}, SeqLen={seqLen}, InFeatures={inFeatures}, Hidden={hiddenSize}");

            // 3. Создаем входной тензор X и заполняем контролируемыми тестовыми данными
            var xShape = new int[] { batchSize, seqLen, inFeatures };
            float[] hostX = new float[batchSize * seqLen * inFeatures];

            // Заполняем простым паттерном для легкой верификации глазами
            for (int i = 0; i < hostX.Length; i++)
                hostX[i] = (float)Math.Sin(i + 1.0); // Детерминированный шум

            Tensor tensorX = new Tensor(hostX, xShape, openClContext, MemoryFlags.ReadWrite, requiresGrad: true);
            
            // 4. Инициализируем слой LSTM
            LSTM lstmLayer = new LSTM(computeContext, nnClGlob, inFeatures, hiddenSize);

            Console.WriteLine("\n[Инициализация параметров слоев во VRAM]:");
            PrintTensorMetadata("WeightW", lstmLayer.WeightW);
            PrintTensorMetadata("WeightU", lstmLayer.WeightU);
            PrintTensorMetadata("Bias", lstmLayer.Bias);

            // =======================================================================
            // ПРЯМОЙ ПРОХОД (FORWARD PASS)
            // =======================================================================
            Console.WriteLine("\n-------------------------------------------------");
            Console.WriteLine("ЗАПУСК: Forward Pass на GPU...");

            Tensor outputH = lstmLayer.Forward(tensorX);

            Console.WriteLine("[Успех] Выходной тензор H успешно вычислен!");
            PrintTensorMetadata("Output H", outputH);

            // Считаем данные Forward на хост для визуальной отладки
            float[] forwardData = DownloadTensorData(outputH, activeQueue);
            Console.WriteLine("Данные Output H (Первые 6 элементов):");
            for (int i = 0; i < Math.Min(6, forwardData.Length); i++)
                Console.Write($"[{forwardData[i]:F4}] ");
            Console.WriteLine("\n-------------------------------------------------");

            // =======================================================================
            // ИМИТАЦИЯ LOSS И ОБРАТНЫЙ ПРОХОД (BACKWARD PASS)
            // =======================================================================
            Console.WriteLine("\nЗАПУСК: Расчет фиктивного градиента (Loss) и BPTT...");

            // В реальном обучении здесь будет LossOp. 
            // Для теста мы сымитируем, что сверху (от функции потерь) пришел градиент dL/dH, 
            // заполненный единицами или контролируемым шумом.
            outputH.InitGradient(openClContext);
            float[] hostGradOutput = new float[outputH.Size];
            for (int i = 0; i < hostGradOutput.Length; i++)
                hostGradOutput[i] = 1.0f; // Заполняем единицами: dL/dH = 1.0

            unsafe
            {
                fixed (float* hgo = hostGradOutput)
                {
                    long bytesGrad = (long)outputH.Size * sizeof(float);
                    activeQueue.EnqueueWriteBuffer(outputH.Grad.DataBuffer, true, 0, bytesGrad, (IntPtr)hgo);
                }
            }

            Console.WriteLine("[Backward] Каскадный обратный проход графа через AutoGrad...");

            // Запускаем ваш движок автоматического дифференцирования
            // Так как у нас нет честного скаляра Loss, мы передаем outputH, 
            // чьи градиенты мы только что принудительно заполнили единицами на GPU.
            AutoGrad.Backward(outputH, openClContext);

            // =======================================================================
            // АНАЛИЗ РЕЗУЛЬТАТОВ И ПРОВЕРКА ГРАДИЕНТОВ
            // =======================================================================
            Console.WriteLine("\n[Анализ буферов градиентов после сцепки BPTT]:");

            bool gradWCreated = lstmLayer.WeightW.Grad != null;
            bool gradUCreated = lstmLayer.WeightU.Grad != null;
            bool gradXCreated = tensorX.Grad != null;

            Console.WriteLine($"-> Градиент для входных весов WeightW создан: {gradWCreated}");
            Console.WriteLine($"-> Градиент для рекуррентных весов WeightU создан: {gradUCreated}");
            Console.WriteLine($"-> Градиент дотек до самого первого входа X: {gradXCreated}");

            if (gradXCreated)
            {
                float[] gradXData = DownloadTensorData(tensorX.Grad, activeQueue);
                Console.WriteLine("\n[Успех] Градиент входа X (первые 4 эл.):");
                for (int i = 0; i < Math.Min(4, gradXData.Length); i++)
                    Console.Write($"[{gradXData[i]:E4}] ");
                Console.WriteLine();
            }

            if (gradWCreated)
            {
                float[] gradWData = DownloadTensorData(lstmLayer.WeightW.Grad, activeQueue);
                Console.WriteLine("[Успех] Градиент весов WeightW (первые 4 эл.):");
                for (int i = 0; i < Math.Min(4, gradWData.Length); i++)
                    Console.Write($"[{gradWData[i]:E4}] ");
                Console.WriteLine();
            }

            Console.WriteLine("\n=== ТЕСТ ОБУЧАЕМОГО СЛОЯ LSTM С УСПЕХОМ ЗАВЕРШЕН ===\n");
        }

        // Вспомогательный метод для вывода метаданных в вашем стиле
        private static void PrintTensorMetadata(string name, Tensor tensor)
        {
            Console.WriteLine($"ОТЛАДКА ТЕНЗОРА: {name}");
            Console.WriteLine($"  Форма (Shape): [{string.Join(", ", tensor.Shape)}]");
            Console.WriteLine($"  Всего элементов: {tensor.Size}");
            Console.WriteLine($"  Требует градиент (RequiresGrad): {tensor.RequiresGrad}");
        }

        // Вспомогательный метод для скачивания данных из VRAM на Хост (CPU)
        private static float[] DownloadTensorData(Tensor tensor, CommandQueue queue)
        {
            float[] hostBuffer = new float[tensor.Size];
            long sizeInBytes = (long)tensor.Size * sizeof(float);
            unsafe
            {
                fixed (float* hb = hostBuffer)
                {
                    queue.EnqueueReadBuffer(tensor.DataBuffer, true, 0, sizeInBytes, (IntPtr)hb);
                }
            }
            queue.Finish();
            return hostBuffer;
        }
     public   static unsafe void TestTrain()
        {
            Console.WriteLine("=== СТАРТ ТЕСТИРОВАНИЯ ОБУЧАЕМОСТИ СЕТИ С LSTM И ADAMW ===");

            // 1. Инициализируем контекст OpenCL (с приоритетом GPU)
            using var computeCtx = new ComputeContext(preferAccelerator: true);
            var context = computeCtx.Context;

            // Регистрируем очередь выполнения в глобальном диспетчере
            DeviceQueueRegistry.RegisterCurrent(context.CreateCommandQueue(computeCtx.Device));
            var queue = DeviceQueueRegistry.GetCurrent();

            // Глобальный диспетчер операций
            NnClGlob clGlob = new NnClGlob(computeCtx);

            // 2. Конфигурация размерностей для сквозного обучения
            int batchSize = 1;      // Обучаем на одном длинном паттерне для простоты
            int seqLen = 5;         // Длина временной последовательности (T)
            int inFeatures = 2;     // Размерность входного вектора
            int hiddenSize = 2;     // Скрытая размерность (равна таргету для прямого MSE)

            int[] xShape = { batchSize, seqLen, inFeatures };
            int[] targetShape = { batchSize, seqLen, hiddenSize }; // [1, 5, 2]

            // 3. Подготовка демонстрационных синтетических данных на CPU
            // Входные данные X: синусоидальное смещение по времени
            float[] inputsData = new float[batchSize * seqLen * inFeatures];
            // Идеальный Target Y: косинусоида (сдвинутая по фазе волна), к которой LSTM должна приблизиться
            float[] targetData = new float[batchSize * seqLen * hiddenSize];

            for (int t = 0; t < seqLen; t++)
            {
                // Наполнение Входа X
                inputsData[t * inFeatures + 0] = (float)Math.Sin(t);
                inputsData[t * inFeatures + 1] = (float)Math.Sin(t + 1);

                // Наполнение Цели Target (к чему сеть ОБЯЗАНА стремиться)
                targetData[t * hiddenSize + 0] = (float)Math.Cos(t);
                targetData[t * hiddenSize + 1] = (float)Math.Cos(t + 1);
            }

            // 4. Аллокация тензоров данных во VRAM
            Tensor inputX = new Tensor(inputsData, xShape, context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad: !false);
            Tensor target = new Tensor(targetData, targetShape, context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad: !false);

            // 5. Инициализируем наш высокоуровневый слой LSTM
            LSTM lstmLayer = new LSTM(computeCtx, clGlob, inFeatures, hiddenSize);

            // Собираем обучаемые параметры для оптимизатора AdamW
            List<Tensor> paramsToOptimize = new List<Tensor>
    {
        lstmLayer.WeightW,
        lstmLayer.WeightU,
        lstmLayer.Bias
    };

            // Создаем оптимизатор с высоким Learning Rate (0.1), чтобы динамика сходимости была видна за 10 эпох
            Adam optimizer = new Adam(paramsToOptimize, lr: 0.1f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f);

            Console.WriteLine("Синтетические датасеты X и Target успешно загружены во VRAM.");
            Console.WriteLine("Запуск оптимизационного цикла (10 эпох)...");
            Console.WriteLine("--------------------------------------------");

            // Буфер для выкачивания значения ошибки на CPU
            float[] lossHost = new float[1];

            // 6. Обучающий цикл (10 эпох)
            for (int epoch = 1; epoch <= 10; epoch++)
            {
                // Шаг 1: Обнуление градиентов весов слоя перед новым проходом
                //lstmLayer.ZeroGrad();
                optimizer.ZeroGrad();
                // Шаг 2: Прямой проход (Forward) через монолитное OpenCL ядро LSTM
                Tensor outputH = lstmLayer.Forward(inputX);

                // Шаг 3: Вычисление ошибки MSE между предсказанием H и идеальным Таргетом
                Tensor loss = clGlob.MseLoss(outputH, target);

                // Синхронно скачиваем значение лосса для мониторинга
                unsafe
                {
                    fixed (float* pL = lossHost)
                    {
                        queue.EnqueueReadBuffer(loss.DataBuffer, true, 0, sizeof(float), (IntPtr)pL);
                    }
                }
                queue.Finish();

                Console.WriteLine($"Эпоха {epoch:D2} | Текущий MSE Loss: {lossHost[0]:F6}");

                // Шаг 4: Обратный проход (Backward) — BPTT разворачивает граф от точки Loss
                AutoGrad.Backward(loss, context);

                // Шаг 5: Оптимизационный шаг — применение ядра AdamW к накопленным градиентам параметров
                optimizer.Step();
            }

            Console.WriteLine("--------------------------------------------");
            Console.WriteLine("Обучение завершено. Проверяем финальное схождение на GPU...");

            // 7. Сравнительный анализ: делаем финальный инференс без расчета градиентов
            lstmLayer.WeightW.RequiresGrad = false; // Отключаем граф для чистоты проверки вывода
            Tensor finalOutputH = lstmLayer.Forward(inputX);

            float[] trainedOutputs = new float[finalOutputH.Size];
            unsafe
            {
                fixed (float* pOut = trainedOutputs)
                {
                    queue.EnqueueReadBuffer(finalOutputH.DataBuffer, true, 0, trainedOutputs.Length * sizeof(float), (IntPtr)pOut);
                }
            }
            queue.Finish();

            Console.WriteLine("\nСравнение предсказаний обученной LSTM (Target vs Trained output):");
            for (int i = 0; i < trainedOutputs.Length; i++)
            {
                Console.WriteLine($"Элемент [{i}]: Идеал (Target) = {targetData[i]:F4} | Выход LSTM = {trainedOutputs[i]:F4}");
            }

            Console.WriteLine("\n=== ТЕСТ ОБУЧАЕМОСТИ LSTM УСПЕШНО ЗАВЕРШЕН ===");
        }
    }
    class Program
    {
        static void Main(string[] args) => LstmDebugger.TestTrain();
    }


namespace cstorchcl.tests
    {
        public static class LstmStatefulDebugger
        {
            public static void RunStatefulTest()
            {
                Console.WriteLine("=== ЗАПУСК ТЕСТА АВТОНОМНОГО СЛОЯ LSTMSTATEFUL ===");

                // 1. Инициализация OpenCL контекста и очередей
               

                // Предполагаем, что ComputeContext и clGlob доступны в вашем тест-раннере
                var computeCtx = new ComputeContext();
                var clGlob = new NnClGlob(computeCtx);
                DeviceQueueRegistry.RegisterCurrent(computeCtx.Context.CreateCommandQueue(computeCtx.Device));
 CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
                Context openClContext = activeQueue.Context;
                // Разномерности конфигурации
                int batchSize = 1;    // Жестко фиксированный размер батча для Stateful слоя
                int seqLen = 2;       // Короткие чанки по 2 шага времени
                int inFeatures = 2;
                int hiddenSize = 2;

                var xShape = new int[] { batchSize, seqLen, inFeatures };

                // 2. Подготовка детерминированных данных для Чанка 1 и Чанка 2
                float[] chunk1Data = new float[] { 0.1f, 0.2f, 0.3f, 0.4f }; // T=0, T=1
                float[] chunk2Data = new float[] { 0.5f, 0.6f, 0.7f, 0.8f }; // T=2, T=3

                // Целевой паттерн (Target) — считаем Loss только для вывода Чанка 2
                float[] targetData = new float[] { 0.9f, 0.9f, 0.9f, 0.9f };

                // Аллокация тензоров данных во VRAM
                Tensor tensorX1 = new Tensor(chunk1Data, xShape, openClContext, MemoryFlags.ReadWrite, requiresGrad: true);
                Tensor tensorX2 = new Tensor(chunk2Data, xShape, openClContext, MemoryFlags.ReadWrite, requiresGrad: true);
                Tensor target2 = new Tensor(targetData, xShape, openClContext, MemoryFlags.ReadWrite, requiresGrad: false);

                // 3. Создание слоя LSTMStateful с явной фиксацией batchSize в конструкторе
                LSTMStateful lstmStateful = new LSTMStateful(computeCtx, clGlob, batchSize, inFeatures, hiddenSize);

                // Гарантируем, что внутренние скрытые состояния сброшены в ноль перед новой сессией
                lstmStateful.ResetStates();

                // =======================================================================
                // ПРЯМОЙ ПРОХОД (FORWARD PASS) ПО ЧАНКАМ
                // =======================================================================
                Console.WriteLine("\n[ШАГ 1]: Прямой проход Чанка 1...");
                // Состояния h и c обновляются и сохраняются внутри слоя автоматически!
                Tensor hOut1 = lstmStateful.Forward(tensorX1);
                Console.WriteLine("-> Чанк 1 успешно обработан слоем.");

                Console.WriteLine("\n[ШАГ 2]: Прямой проход Чанка 2...");
                // Слой использует сохраненный внутренний контекст от Чанка 1 и связывает граф
                Tensor hOut2 = lstmStateful.Forward(tensorX2);
                Console.WriteLine("-> Чанк 2 успешно обработан слоем.");

                // =======================================================================
                // ВЫЧИСЛЕНИЕ LOSS И ОБРАТНЫЙ ПРОХОД (BPTT)
                // =======================================================================
                Console.WriteLine("\n[ШАГ 3]: Вычисление ошибки MSE Loss на выходе Чанка 2...");
                Tensor loss = clGlob.MseLoss(hOut2, target2);

                float[] lossHost = new float[1];
                unsafe
                {
                    fixed (float* pL = lossHost)
                        activeQueue.EnqueueReadBuffer(loss.DataBuffer, true, 0, sizeof(float), (IntPtr)pL);
                }
                Console.WriteLine($"-> Текущий MSE Loss на Чанке 2: {lossHost[0]:F6}");

                Console.WriteLine("\n[ШАГ 4]: Запуск сквозного обратного прохода (AutoGrad.Backward)...");
                // Движок начнет разворачивать граф: 
                // Loss -> Чанк 2 -> Внутренние ссылки на состояния Чанка 1 -> Чанк 1 -> Данные X1
                AutoGrad.Backward(loss, openClContext);
                Console.WriteLine("-> Каскадный обход BPTT успешно завершен.");

                // =======================================================================
                // АНАЛИЗ ГРАДИЕНТОВ
                // =======================================================================
                Console.WriteLine("\n=== АНАЛИЗ ПЕРЕТЕКАНИЯ ГРАДИЕНТОВ В LSTMSTATEFUL ===");

                bool gradX2Created = tensorX2.Grad != null;
                bool gradX1Created = tensorX1.Grad != null;
                bool gradWCreated = lstmStateful.WeightW.Grad != null;

                Console.WriteLine($"1. Градиент дотек до данных Чанка 2 (gradX2): {gradX2Created}");
                Console.WriteLine($"2. Градиент перешагнул барьер чанков и дотек до Чанка 1 (gradX1): {gradX1Created}");
                Console.WriteLine($"3. Градиент для объединенных весов WeightW создан: {gradWCreated}");

                if (gradX1Created)
                {
                    float[] x1Grads = DownloadTensorData(tensorX1.Grad, activeQueue);
                    Console.WriteLine($"\n[Успех] Значение dL/dX1 (первый элемент Чанка 1): {x1Grads[0]:E4}");
                }

                if (gradWCreated)
                {
                    float[] wGrads = DownloadTensorData(lstmStateful.WeightW.Grad, activeQueue);
                    Console.WriteLine($"[Успех] Значение dL/dW (первый элемент матрицы весов): {wGrads[0]:E4}");
                }

                Console.WriteLine("\n=== ТЕСТ LSTMSTATEFUL УСПЕШНО ЗАВЕРШЕН ===\n");
            }

            private static float[] DownloadTensorData(Tensor tensor, CommandQueue queue)
            {
                float[] hostBuffer = new float[tensor.Size];
                long sizeInBytes = (long)tensor.Size * sizeof(float);
                unsafe
                {
                    fixed (float* hb = hostBuffer)
                    {
                        queue.EnqueueReadBuffer(tensor.DataBuffer, true, 0, sizeInBytes, (IntPtr)hb);
                    }
                }
                queue.Finish();
                return hostBuffer;
            }
        }
    }

}
