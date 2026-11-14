using cstorchcl.compiler;
using cstorchcl.core;
using cstorchcl.optimizers;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TestLSTMRealTrain;

namespace cstorchcl.tests
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== ЗАПУСК КОРРЕКТНОГО ТЕСТА ENCODER-DECODER ===");

            string datasetPath = @"C:\LLModels\Gemma4_E2B\fulldataset.bin";
            datasetPath = "fulldataset.bin";
            int vocabSize = 6565;
            int encoderHiddenDim = 500;
            int batchSize = 8;
            int blockTokens = 1500;

            using var computeCtx = new ComputeContext(preferAccelerator: true);
            var context = computeCtx.Context;
            DeviceQueueRegistry.RegisterCurrent(context.CreateCommandQueue(computeCtx.Device));
            var queue = DeviceQueueRegistry.GetCurrent();
            NnClGlob clGlob = new NnClGlob(computeCtx);

            // Инициализация модели
            var model = new EncoderDecoderNetwork(computeCtx, clGlob, encoderHiddenDim, batchSize);

            // ИСПРАВЛЕНИЕ 1: Создаем тензоры весов эмбеддингов прямо в коде теста
            int[] embShape = { vocabSize, encoderHiddenDim };
            Tensor embeddingWeightsEncoder = new Tensor(embShape, context,"embed_encoder_weight", MemoryFlags.ReadWrite, requiresGrad: true);
            Tensor embeddingWeightsDecoder = new Tensor(embShape, context,"embed_decoder_weigth", MemoryFlags.ReadWrite, requiresGrad: true);
            float[] initW = new float[vocabSize * encoderHiddenDim];
            Random r = new Random();
            float bound = (float)(Math.Sqrt(6.0) / Math.Sqrt(vocabSize));
            for (int k = 0; k < initW.Length; k++)
                initW[k] = (float)(r.NextDouble() * 2.0 - 1.0) * bound;
            // Заполняем эмбеддинги случайным шумом (Xavier/Glorot для эмбеддингов)
            // ... (код заполнения опущен для краткости)
            unsafe
            {
                fixed (float* _t = initW)
                {
                    queue.EnqueueWriteBuffer(embeddingWeightsEncoder.DataBuffer, true, 0, sizeof(float) * initW.Length, (IntPtr)_t);
                    queue.EnqueueWriteBuffer(embeddingWeightsDecoder.DataBuffer, true, 0, sizeof(float) * initW.Length, (IntPtr)_t);
                    queue.Finish();
                }
            }
            // Настраиваем список оптимизатора параметров
            List<Tensor> parameters = new List<Tensor>();

            // ДОБАВЛЯЕМ ТЕНЗОРЫ ЭМБЕДДИНГОВ В ОПТИМИЗАТОР
            parameters.Add(embeddingWeightsEncoder);
            parameters.Add(embeddingWeightsDecoder);

            // Добавляем веса остальных слоев
            parameters.AddRange(model.EncoderLstm.GetParameters());
            parameters.AddRange(model.DecoderLstm.GetParameters());
            parameters.AddRange(model.FullyConnectedOut.GetParameters());

                //mockLossTarget.Dispose();
            Adam optimizer = new Adam(computeCtx,parameters, lr: 1e-4f);

            using var dataLoader = new StreamDataLoader(datasetPath, batchSize, blockTokens,context,queue.Device);

            Console.WriteLine("\n[Успех] Все тензоры параметров внесены в Adam.");
            Console.WriteLine("----------------------------------------------------------------");
            //var encInputBuf = context.CreateBuffer(MemoryFlags.ReadOnly, sizeof(int) * blockTokens * batchSize);
            //var decTargetBuf= context.CreateBuffer(MemoryFlags.ReadOnly, sizeof(int) * blockTokens * batchSize);
            int BATCH_COUNT = 7;
            double sred_inter = 0;
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
                Tensor logits = model.Forward(encInputBuf, decTargetBuf, embeddingWeightsEncoder, embeddingWeightsDecoder, blockTokens, batchSize);

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
                IntPtr pL = HostPtrBufferManager.GetOrCreateBuffer("loss1", sizeof(float));
                unsafe
                {
                    //fixed (float* pL = lossValue)

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
