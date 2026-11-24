using cstorchcl.compiler;
using cstorchcl.core;
using cstorchcl.layers;
using cstorchcl.optimizers;
using cstorchcl.serialization;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TestLSTMRealTrain;

namespace cstorchcl.tests
{
    public class Program
    {
       
        private unsafe static void Main2()
        {
  const int BLOCK_TOKENS = 1500;
         const int BLOCK_BYTES = BLOCK_TOKENS * sizeof(int); // 6000 байт
            Console.WriteLine("=== ОБУЧЕНИЕ DECODER-ONLY LLM (ПОТОКОВОЕ ЧТЕНИЕ > 2 ГБ) ===");

            string datasetPath = "fulldataset.bin";
            if (!File.Exists(datasetPath))
            {
                Console.WriteLine($"Файл датасета не найден: {datasetPath}");
                return;
            }

            using var computeCtx = new ComputeContext(preferAccelerator: true);
            using var queue = computeCtx.Context.CreateCommandQueue(computeCtx.Device);
            DeviceQueueRegistry.RegisterCurrent(queue);
            var nnClGlob = new NnClGlob(computeCtx);

            // Гиперпараметры модели
            int vocabSize = 6565;
            int dModel = 256;
            int numLayers = 4;
            int numHeads = 4;          // head_dim = 64
            int intermediateDim = 688; // ~ 8/3 * 256
            int batchSize = 8;
            int seqLen = 128;          // Должно быть строго меньше BLOCK_TOKENS (1500)

            if (seqLen + 1 > BLOCK_TOKENS)
                throw new ArgumentException($"seqLen + 1 ({seqLen + 1}) не может превышать размер блока ({BLOCK_TOKENS}).");

            var model = new DecoderLLM(
                computeCtx,
                nnClGlob,
                vocabSize: vocabSize,
                dModel: dModel,
                numLayers: numLayers,
                numHeads: numHeads,
                intermediateDim: intermediateDim,
                modelId: "gemma_mini");

            var parameters = model.GetParameters().ToList();
            long totalWeights = parameters.Sum(p => (long)p.Size);
            Console.WriteLine($"Модель создана: {numLayers} слоев | Всего весов: {totalWeights:N0}");

            var optimizer = new Adam(computeCtx,parameters,  lr: 0.0003f);

            // GPU-буферы под входные и целевые токены [Batch * SeqLen]
            long bufBytes = (long)batchSize * seqLen * sizeof(int);
            using NOpenCL.Buffer inputBuf = computeCtx.Context.CreateBuffer(MemoryFlags.ReadOnly, bufBytes);
            using NOpenCL.Buffer targetBuf = computeCtx.Context.CreateBuffer(MemoryFlags.ReadOnly, bufBytes);

            int[] hostInputs = new int[batchSize * seqLen];
            int[] hostTargets = new int[batchSize * seqLen];

            // Временные буферы на CPU для чтения одного окна (seqLen + 1 токенов) без аллокаций в цикле
            int windowTokensCount = seqLen + 1;
            int windowBytesCount = windowTokensCount * sizeof(int);
            byte[] rawWindowBytes = new byte[windowBytesCount];
            int[] windowTokens = new int[windowTokensCount];

            // Открываем поток с флагом RandomAccess для быстрого Seek по большим файлам (> 2 ГБ)
            using var fs = new FileStream(
                datasetPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 65536,
                options: FileOptions.RandomAccess);

            using var reader = new BinaryReader(fs);

            long totalBlocks = fs.Length / BLOCK_BYTES;
            long totalDatasetTokens = totalBlocks * BLOCK_TOKENS;
            Console.WriteLine($"Файл: {fs.Length / (1024.0 * 1024.0 * 1024.0):F2} ГБ | Блоков по 1500: {totalBlocks:N0} | Всего токенов: {totalDatasetTokens:N0}");

            var rand = new Random(123);
            int totalSteps = 1000;

            for (int step = 1; step <= totalSteps; step++)
            {
                Stopwatch sw = Stopwatch.StartNew();

                // 1. Формируем батч чтением случайных окон через Seek
                for (int b = 0; b < batchSize; b++)
                {
                    // Выбираем случайный блок (64-битный индекс через rand.NextInt64)
                    long blockIdx = rand.NextInt64(0, totalBlocks);
                    // Выбираем случайное смещение внутри этого блока так, чтобы окно seqLen + 1 не вышло за границу 1500
                    int tokenOffsetInBlock = rand.Next(0, BLOCK_TOKENS - windowTokensCount + 1);

                    // Вычисляем точное 64-битное смещение в байтах от начала файла
                    long byteOffset = (blockIdx * BLOCK_BYTES) + ((long)tokenOffsetInBlock * sizeof(int));

                    fs.Seek(byteOffset, SeekOrigin.Begin);
                    int bytesRead = reader.Read(rawWindowBytes, 0, windowBytesCount);
                    if (bytesRead < windowBytesCount)
                        throw new EndOfStreamException($"Неожиданный конец файла при чтении блока {blockIdx}.");

                    // Копируем байты в массив int32
                    System.Buffer.BlockCopy(rawWindowBytes, 0, windowTokens, 0, windowBytesCount);

                    // Заполняем input [0 .. seqLen-1] и сдвинутый на +1 target [1 .. seqLen]
                    int batchOffset = b * seqLen;
                    for (int t = 0; t < seqLen; t++)
                    {
                        hostInputs[batchOffset + t] = windowTokens[t];
                        hostTargets[batchOffset + t] = windowTokens[t + 1];
                    }
                }

                // 2. Копируем батч на видеокарту
                fixed (int* pIn = hostInputs)
                fixed (int* pTgt = hostTargets)
                {
                    queue.EnqueueWriteBuffer(inputBuf, true, 0, bufBytes, (IntPtr)pIn);
                    queue.EnqueueWriteBuffer(targetBuf, true, 0, bufBytes, (IntPtr)pTgt);
                }

                optimizer.ZeroGrad();

                // 3. Forward -> [Batch, SeqLen, VocabSize]
                Tensor logits = model.Forward(inputBuf, batchSize, seqLen);

                // Если ваша SparseCrossEntropyLoss ожидает 2D-тензор [Batch * SeqLen, VocabSize], раскомментируйте строку ниже:
                // logits.ReShape(new[] { batchSize * seqLen, vocabSize });

                // 4. Cross-Entropy Loss
                Tensor loss = nnClGlob.SparseCrossEntropyLoss(logits, targetBuf);
                float[] lossVal = HostReturner.ToCpu(loss);

                // 5. Backward + шаг оптимизатора Adam
                AutoGrad.Backward(loss, computeCtx.Context);
                optimizer.Step();
                queue.Finish();
                sw.Stop();

                if (step % 10 == 0 || step == 1)
                {
                    double tokPerSec = (batchSize * seqLen) / sw.Elapsed.TotalSeconds;
                    Console.WriteLine($"Шаг {step:D4}/{totalSteps} | Loss: {lossVal[0]:F4} | Время: {sw.Elapsed.TotalMilliseconds:F1} мс | {tokPerSec:F0} ток/с");
                }
            }

            // Сохраняем веса модели в safetensors
            var meta = new Dictionary<string, string>
            {
                { "arch", "decoder_llm_rope" },
                { "vocab_size", vocabSize.ToString() },
                { "d_model", dModel.ToString() },
                { "num_layers", numLayers.ToString() },
                { "num_heads", numHeads.ToString() }
            };
            SafeTensorsSerializer.Save("decoder_llm.safetensors", parameters, meta);
            Console.WriteLine("Обучение завершено, веса сохранены в decoder_llm.safetensors!");
        }
        public static void Main(string[] args)
        {
            Main2();
            return;
            Console.WriteLine("=== ЗАПУСК КОРРЕКТНОГО ТЕСТА ENCODER-DECODER ===");

            string datasetPath = @"C:\LLModels\Gemma4_E2B\fulldataset.bin";
            datasetPath = "fulldataset.bin";
            int vocabSize = 6565;
            int encoderHiddenDim = 500;
            int batchSize = 8;
            int blockTokens = 1500;
            int PRINT_COUNT = 5;
            using var computeCtx = new ComputeContext(preferAccelerator: true);
            var context = computeCtx.Context;
            DeviceQueueRegistry.RegisterCurrent(context.CreateCommandQueue(computeCtx.Device));
            var queue = DeviceQueueRegistry.GetCurrent();
            NnClGlob clGlob = new NnClGlob(computeCtx);

            // Инициализация модели
            var model = new EncoderDecoderNetwork(computeCtx, clGlob, encoderHiddenDim, batchSize,vocabSize);

            // ИСПРАВЛЕНИЕ 1: Создаем тензоры весов эмбеддингов прямо в коде теста
            int[] embShape = { vocabSize, encoderHiddenDim };
            //Tensor embeddingWeightsEncoder = new Tensor(embShape, context,"embed_encoder_weight", MemoryFlags.ReadWrite, requiresGrad: true);
           // Tensor embeddingWeightsDecoder = new Tensor(embShape, context,"embed_decoder_weigth", MemoryFlags.ReadWrite, requiresGrad: true);
            float[] initW = new float[vocabSize * encoderHiddenDim];
            Random r = new Random();
            float bound = (float)(Math.Sqrt(6.0) / Math.Sqrt(vocabSize));
            for (int k = 0; k < initW.Length; k++)
                initW[k] = (float)(r.NextDouble() * 2.0 - 1.0) * bound;
            // Заполняем эмбеддинги случайным шумом (Xavier/Glorot для эмбеддингов)
            // ... (код заполнения опущен для краткости)
            unsafe
            {
               /* fixed (float* _t = initW)
                {
                    queue.EnqueueWriteBuffer(embeddingWeightsEncoder.DataBuffer, true, 0, sizeof(float) * initW.Length, (IntPtr)_t);
                    queue.EnqueueWriteBuffer(embeddingWeightsDecoder.DataBuffer, true, 0, sizeof(float) * initW.Length, (IntPtr)_t);
                    queue.Finish();
                }*/
            }
            // Настраиваем список оптимизатора параметров
            List<Tensor> parameters = new List<Tensor>();

            // ДОБАВЛЯЕМ ТЕНЗОРЫ ЭМБЕДДИНГОВ В ОПТИМИЗАТОР
            // parameters.Add(embeddingWeightsEncoder);
            // parameters.Add(embeddingWeightsDecoder);

            // Добавляем веса остальных слоев
            parameters.AddRange(model.GetParameters());
            parameters.AddRange(model.EncoderLstm.GetParameters());
            parameters.AddRange(model.DecoderLstm.GetParameters());
            parameters.AddRange(model.FullyConnectedOut.GetParameters());

                //mockLossTarget.Dispose();
            Adam optimizer = new Adam(computeCtx,parameters, lr: 1e-3f);

            using var dataLoader = new StreamDataLoader(datasetPath, batchSize, blockTokens,context,queue.Device);

            Console.WriteLine("\n[Успех] Все тензоры параметров внесены в Adam.");
            Console.WriteLine("----------------------------------------------------------------");
            //var encInputBuf = context.CreateBuffer(MemoryFlags.ReadOnly, sizeof(int) * blockTokens * batchSize);
            //var decTargetBuf= context.CreateBuffer(MemoryFlags.ReadOnly, sizeof(int) * blockTokens * batchSize);
            int BATCH_COUNT = 500;
            double sred_inter = 0;
            IntPtr pL = HostPtrBufferManager.GetOrCreateBuffer("loss1", sizeof(float));
            for (int batchIdx = 0; batchIdx < BATCH_COUNT; batchIdx++)
            {
                Stopwatch batchTimer = Stopwatch.StartNew();

                optimizer.ZeroGrad();
                //model.DecoderLstm.ResetStates();

                // Извлекаем нативные буферы из даталоадера
                //dataLoader.GetBatch(batchIdx, context,encInputBuf,decTargetBuf);
                var (encInputBuf, decTargetBuf) = dataLoader.GetBatch(batchIdx);
                // ... [Внутри обучающего цикла] ...

                // Прямой проход сети — получаем полные логиты [Batch, SeqLen, VocabSize]
                Tensor logits = model.Forward(encInputBuf, decTargetBuf,  blockTokens, batchSize);

                // Подготавливаем One-Hot мишень со сдвигом [Batch, SeqLen - 1, VocabSize] на основе первого ядра
                //Tensor trueLossTarget = clGlob.PrepareTrueTargets(decTargetBuf, batchSize, blockTokens, vocabSize);

                // Усекаем логиты до (SeqLen - 1), и теперь SliceOp автоматически свяжет граф!
                //Tensor slicedLogits = clGlob.SliceLogitsForLoss(logits,computeCtx);

                // Считаем честную ошибку между усеченными логитами и One-Hot таргетом
                /*Console.WriteLine("тензор slicedLogits");
                HostReturner.PrintTensor(slicedLogits);
                Console.WriteLine("тензор trueLossTarget");
                HostReturner.PrintTensor(trueLossTarget);*/
                //Tensor loss = clGlob.CrossEntropyLoss(slicedLogits, trueLossTarget);
                Tensor loss = clGlob.SparseCrossEntropyLoss(logits, decTargetBuf);
                // ... [Backward и очистка памяти] ...


                //  float[] lossValue = new float[1];
               
                unsafe
                {
                    //fixed (float* pL = lossValue)
                    if((batchIdx+1)%PRINT_COUNT==0)
                        queue.EnqueueReadBuffer(loss.DataBuffer, true, 0, sizeof(float), (IntPtr)pL);
                }
                //queue.Finish();

                AutoGrad.Backward(loss, context);
                optimizer.Step();


                batchTimer.Stop();
                long totalTokensProcessed = (long)batchSize * blockTokens;
                double tokensPerSecond = totalTokensProcessed / batchTimer.Elapsed.TotalSeconds;
                if (batchIdx >= 3||BATCH_COUNT<3)
                {
                    sred_inter += batchTimer.Elapsed.TotalSeconds;
                }
                if ((batchIdx + 1) % PRINT_COUNT== 0)
                    unsafe
                {
                    Console.WriteLine($"Батч {batchIdx + 1:D2}/{BATCH_COUNT} | MSE Loss: {((float*)pL)[0]:F6} | Скорость: {tokensPerSecond:F0} токенов/сек({1 / batchTimer.Elapsed.TotalSeconds} it/s)");

                }
            }
            queue.Finish();
            sred_inter /= BATCH_COUNT - (BATCH_COUNT>3?3:0);

            //encInputBuf.Dispose();
           // decTargetBuf.Dispose();
            Console.WriteLine("----------------------------------------------------------------");
            Console.WriteLine($"средняя скорость обработки:{1/sred_inter} it/s");
            Console.WriteLine("----------------------------------------------------------------");
            Console.WriteLine("Эксперимент успешно выполнен!");
            // Собираем метаданные (необязательно, но полезно для совместимости с HuggingFace)
            var meta = new Dictionary<string, string>
{
    { "format", "pt" },
    { "framework", "cstorchcl" },
    { "vocab_size", vocabSize.ToString() },
    { "hidden_dim", encoderHiddenDim.ToString() }
};

            // Сохраняем все параметры, которые передавали в Adam
            SafeTensorsSerializer.Save("encoder_decoder_model.safetensors", parameters, meta);
            Console.WriteLine("Веса модели успешно сохранены в safetensors!");

            OpenCLBufferManager.DisposeAll();
            cstorchcl.compiler.HostPtrBufferManager.DisposeAll();
        }
        
        /// <summary>
        /// Расширяет усеченный градиент до исходного размера, заполняя последний шаг времени нулями.
        /// </summary>
        public static void PadSlicedGradient(Tensor gradSliced, Tensor gradOriginal)
        {
            int batchSize = gradOriginal.Shape[0];
            int oldSeqLen = gradOriginal.Shape[1];
            int vocabSize = gradOriginal.Shape[2];
            int newSeqLen = oldSeqLen - 1;

            long rowBytes = (long)vocabSize * sizeof(float);
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Сначала полностью зануляем весь буфер оригинального градиента во VRAM,
            // чтобы последний неиспользуемый шаг гарантированно содержал нули
            float[] zeroPattern = new float[vocabSize]; // массив нулей на CPU размера одной строки
            unsafe
            {
                fixed (float* zp = zeroPattern)
                {
                    for (int b = 0; b < batchSize; b++)
                    {
                        long lastStepOffset = (long)(b * oldSeqLen + newSeqLen) * rowBytes;
                        queue.EnqueueWriteBuffer(gradOriginal.DataBuffer, false, (IntPtr)lastStepOffset, rowBytes, (IntPtr)zp);
                    }
                }
            }

            // Копируем накопленные градиенты из усеченного тензора в оригинальный шаг за шагом по батчу
            for (int b = 0; b < batchSize; b++)
            {
                long srcOffset = (long)b * newSeqLen * rowBytes;
                long destOffset = (long)b * oldSeqLen * rowBytes;
                long bytesToCopy = (long)newSeqLen * rowBytes;

                queue.EnqueueCopyBuffer(gradSliced.DataBuffer, gradOriginal.DataBuffer, (IntPtr)srcOffset, (IntPtr)destOffset, (IntPtr)bytesToCopy);
            }

            queue.Finish();
        }

    }
}
