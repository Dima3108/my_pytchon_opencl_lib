namespace InferenceTest
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using cstorchcl.compiler;
    using cstorchcl.core;
    using cstorchcl.serialization;
    using cstorchcl.tokenizers;
    using NOpenCL;

 
    
        public static class Program
        {
            // Настройте ID служебных токенов под ваш словарь
            private const int SEQ_LEN = 100;
            private static  int PAD_TOKEN_ID = 0;
            private static int BOS_TOKEN_ID = 1; // Токен начала генерации для декодера
            private static int EOS_TOKEN_ID = 2; // Токен конца генерации

            public static unsafe void Main(string[] args)
            {
                Console.OutputEncoding = System.Text.Encoding.UTF8;
                Console.InputEncoding = System.Text.Encoding.UTF8;

                string weightsPath = "encoder_decoder_model.safetensors";
                string vocabPath = "my_vocab.json"; // Путь к JSON-словарю токенизатора

                int vocabSize = 6565;
                int hiddenDim = 500;
                int batchSize = 1; // Инференс по одному запросу

                // 1. Инициализация OpenCL
                using var computeCtx = new ComputeContext(preferAccelerator: true);
                var context = computeCtx.Context;
                DeviceQueueRegistry.RegisterCurrent(context.CreateCommandQueue(computeCtx.Device));
                var queue = DeviceQueueRegistry.GetCurrent();
                var clGlob = new NnClGlob(computeCtx);

                // 2. Инициализация токенизатора
                var tokenizer = new Tokenizer256();
                if (File.Exists(vocabPath))
                {
                    tokenizer.load_from_json(vocabPath);
                    Console.WriteLine($"[OK] Словарь загружен из {vocabPath}");
                }
                else
                {
                    Console.WriteLine("[Внимание] Файл словаря не найден, используются базовые 256 байт-токенов.");
                }
            PAD_TOKEN_ID = tokenizer.encode("[PAD]")[0];
            EOS_TOKEN_ID = tokenizer.encode("[EOS]")[0];
            BOS_TOKEN_ID = tokenizer.encode("[BOS]")[0];
                // 3. Создание архитектуры модели и тензоров эмбеддингов
                var model = new EncoderDecoderNetwork(computeCtx, clGlob, hiddenDim, batchSize,vocabSize);

                int[] embShape = { vocabSize, hiddenDim };
                // Имена TensorID должны в точности совпадать с теми, что были при сохранении
               // var embeddingWeightsEncoder = new Tensor(embShape, context, "embed_encoder_weight", MemoryFlags.ReadWrite, requiresGrad: false);
               // var embeddingWeightsDecoder = new Tensor(embShape, context, "embed_decoder_weigth", MemoryFlags.ReadWrite, requiresGrad: false);

            var allParameters = new List<Tensor>();
            allParameters.AddRange(model.GetParameters());
                allParameters.AddRange(model.EncoderLstm.GetParameters());
                allParameters.AddRange(model.DecoderLstm.GetParameters());
                allParameters.AddRange(model.FullyConnectedOut.GetParameters());

                // Отключаем расчет градиентов для инференса (освобождает workspaceBuf в LSTM)
                foreach (var p in allParameters)
                {
                    p.RequiresGrad = false;
                }

                // 4. Загрузка весов из safetensors в VRAM
                var metadata = SafeTensorsSerializer.Load(weightsPath, allParameters);
                Console.WriteLine($"[OK] Веса успешно загружены из {weightsPath}");
                foreach (var kv in metadata)
                {
                    Console.WriteLine($"     meta -> {kv.Key}: {kv.Value}");
                }

                // 5. Выделяем GPU-буферы под вход энкодера и вход декодера на 100 токенов
                long inputBufferBytes = sizeof(int) * SEQ_LEN * batchSize;
                using NOpenCL.Buffer encInputBuf = context.CreateBuffer(MemoryFlags.ReadOnly, inputBufferBytes);
                using NOpenCL.Buffer decInputBuf = context.CreateBuffer(MemoryFlags.ReadOnly, inputBufferBytes);

                float[] stepLogits = new float[vocabSize];

                Console.WriteLine("\nВведите текст для обработки (или 'exit' для выхода):");
                while (true)
                {
                    Console.Write("\n> ");
                    string? inputText = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(inputText) || inputText.Equals("exit", StringComparison.OrdinalIgnoreCase))
                        break;

                    // Токенизируем с ограничением не более 100 токенов
                    List<int> tokens = tokenizer.encode(inputText, SEQ_LEN);
                    int realTokenCount = tokens.Count;

                    // Формируем массив длиной 100 и дополняем PAD-токенами при необходимости
                    int[] encoderTokens = new int[SEQ_LEN];
                    for (int i = 0; i < SEQ_LEN; i++)
                    {
                        encoderTokens[i] = i < realTokenCount ? tokens[i] : PAD_TOKEN_ID;
                    }

                    Console.WriteLine($"[Токенов: {realTokenCount}, дополнено PAD до {SEQ_LEN}]");

                    // Копируем вход энкодера на GPU
                    fixed (int* pEnc = encoderTokens)
                    {
                        queue.EnqueueWriteBuffer(encInputBuf, true, 0, inputBufferBytes, (IntPtr)pEnc);
                    }

                    // Подготавливаем буфер декодера: [BOS, PAD, PAD, ..., PAD]
                    int[] decoderTokens = new int[SEQ_LEN];
                    Array.Fill(decoderTokens, PAD_TOKEN_ID);
                    decoderTokens[0] = BOS_TOKEN_ID;

                    var generatedIds = new List<int>();

                    // Пошаговая авторегрессионная генерация
                    for (int step = 0; step < SEQ_LEN - 1; step++)
                    {
                        fixed (int* pDec = decoderTokens)
                        {
                            queue.EnqueueWriteBuffer(decInputBuf, true, 0, inputBufferBytes, (IntPtr)pDec);
                        }

                        // Прямой проход: получаем логиты [1, 100, vocabSize]
                        using Tensor logits = model.Forward(
                            encInputBuf,
                            decInputBuf,
                           // embeddingWeightsEncoder,
                           // embeddingWeightsDecoder,
                            SEQ_LEN,
                            batchSize);

                        // Считываем с видеокарты логиты только для текущего шага `step`
                        long offsetBytes = (long)step * vocabSize * sizeof(float);
                        long sliceBytes = (long)vocabSize * sizeof(float);

                        fixed (float* pLogits = stepLogits)
                        {
                            queue.EnqueueReadBuffer(logits.DataBuffer, true, offsetBytes, sliceBytes, (IntPtr)pLogits);
                        }

                        // Выбираем токен с максимальной вероятностью (Greedy Argmax)
                        int nextTokenId = ArgMax(stepLogits);

                        if (nextTokenId == EOS_TOKEN_ID || nextTokenId == PAD_TOKEN_ID)
                            break;

                        generatedIds.Add(nextTokenId);
                        decoderTokens[step + 1] = nextTokenId;
                    }

                    string outputText = tokenizer.decode(generatedIds.ToArray());
                    Console.WriteLine($"Ответ модели: {outputText}");
                }

                // Очистка ресурсов
                model.Dispose();
                //embeddingWeightsEncoder.Dispose();
              //  embeddingWeightsDecoder.Dispose();
                OpenCLBufferManager.DisposeAll();
                HostPtrBufferManager.DisposeAll();
            }

            private static int ArgMax(float[] array)
            {
                int bestIdx = 0;
                float bestVal = array[0];
                for (int i = 1; i < array.Length; i++)
                {
                    if (array[i] > bestVal)
                    {
                        bestVal = array[i];
                        bestIdx = i;
                    }
                }
                return bestIdx;
            }
        }
    }


