using cstorchcl.compiler;
using cstorchcl.layers;
using NOpenCL;
using System;

namespace cstorchcl.core
{
    public class EncoderDecoderNetwork : NeuralNetwork
    {
        // Слои нашей сети (веса эмбеддингов теперь хранятся на уровне Program/Main)
        public LSTM EncoderLstm { get; private set; }
        public LSTM DecoderLstm { get; private set; }
        public Linear FullyConnectedOut { get; private set; }
        public Tensor encoderWeigth;
        public Tensor decoderWeigth;
        private readonly int _hiddenDim;

        public EncoderDecoderNetwork(ComputeContext computeContext, NnClGlob nnClGlob, int hiddenDim, int batchSize,int vocabsize)
            : base(computeContext, nnClGlob,"encoderdfecodermodel")
        {
            _hiddenDim = hiddenDim;
            encoderWeigth = RegisterParameter(new int[] { vocabsize, hiddenDim }, nameof(encoderWeigth), true);
            decoderWeigth = RegisterParameter(new int[] { vocabsize, hiddenDim }, nameof(decoderWeigth), true);
            // 1. Энкодер: принимает эмбеддинг токена -> выдает hidden_dim
            EncoderLstm = new LSTM(computeContext, nnClGlob, inFeatures: _hiddenDim, hiddenSize: _hiddenDim,"encoderlstm");

            // 2. Декодер: принимает конкатенацию (embedding + encoder_output) -> размерность 2 * hidden_dim
            DecoderLstm = new LSTM(computeContext, nnClGlob, inFeatures: _hiddenDim * 2, hiddenSize: _hiddenDim,"decoderlstm");

            // 3. Финальная проекция в размерность словаря
            FullyConnectedOut = new Linear(computeContext, nnClGlob, inFeatures: _hiddenDim, outFeatures: 6565,layerId:"outLinear"); // VocabSize = 6565
            InitWeight();
        }
        private void InitWeight()
        {
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            var rand = new Random();
            int totalGateFeatures = 4 * encoderWeigth.Shape[1];

            // --- Инициализация WeightW (Xavier) ---
            float boundW = (float)(Math.Sqrt(6.0) / Math.Sqrt(encoderWeigth.Shape[0] + encoderWeigth.Shape[1]));
            float[] hostWeightsW = new float[encoderWeigth.Size];
            for (int i = 0; i < hostWeightsW.Length; i++)
            {
                hostWeightsW[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundW;
            }

            // --- Инициализация WeightU (Xavier) ---
            float boundU = (float)(Math.Sqrt(6.0) / Math.Sqrt(decoderWeigth.Shape[0] + decoderWeigth.Shape[1]));
            float[] hostWeightsU = new float[decoderWeigth.Size];
            for (int i = 0; i < hostWeightsU.Length; i++)
            {
                hostWeightsU[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundU;
            }
            unsafe
            {
                fixed(float* enW = hostWeightsW, decW = hostWeightsU)
                {
                    queue.EnqueueWriteBuffer(encoderWeigth.DataBuffer, true, 0, sizeof(float) * hostWeightsU.Length, (IntPtr)enW);
                    queue.EnqueueWriteBuffer(decoderWeigth.DataBuffer, true, 0, sizeof(float) * hostWeightsU.Length, (IntPtr)decW);
                    queue.Finish();
                }
            }
        }
        /// <summary>
        /// Скорректированный Forward Pass без явного слоя Embedding и без операции Concat.
        /// </summary>
        public Tensor Forward(
            NOpenCL.Buffer encoderInputsBuf,
            NOpenCL.Buffer decoderTargetsBuf,
           // Tensor embWeightsEnc, // Явно принимаем тензоры весов эмбеддингов
           // Tensor embWeightsDec,
            int seqLen,
            int batchSize)
        {
            // === ШАГ 1: ЭНКОДЕР ===
            // Вызываем эмбеддинг через глобальный объект clGlob
            Tensor embeddedEncoder = NnClGlob.Embedding(encoderInputsBuf, encoderWeigth, batchSize * seqLen,LAYER_ID+".embed1");
            //embeddedEncoder.Shape = new int { batchSize, seqLen, _hiddenDim };
            embeddedEncoder.ReShape(new int []{ batchSize, seqLen, _hiddenDim });
            // Прогон через LSTM Энкодера
            Tensor encoderOutputs = EncoderLstm.Forward(embeddedEncoder); // [Batch, SeqLen, HiddenDim]

            // === ШАГ 2: ДЕКОДЕР ===
            Tensor embeddedDecoder = NnClGlob.Embedding(decoderTargetsBuf, decoderWeigth, batchSize * seqLen, LAYER_ID + ".embed2");
            //embeddedDecoder.Shape = new int { batchSize, seqLen, _hiddenDim };
            embeddedDecoder.ReShape(new int[] { batchSize, seqLen, _hiddenDim });

            // Вместо ручной аллокации и псевдокода — вызываем полноценный Cat
            // Поведение полностью идентично torch.cat((embeddedDecoder, encoderOutputs), dim=-1)
#if false
            Console.WriteLine("данные перед конкатенацией:");
            HostReturner.PrintTensor(embeddedDecoder);
            HostReturner.PrintTensor(encoderOutputs);
#endif
            Tensor decoderInputs = NnClGlob.Cat(embeddedDecoder, encoderOutputs, dim: -1);
#if false
            Console.WriteLine("результат cat:");
            HostReturner.PrintTensor(decoderInputs);

#endif
            Tensor decoderOutputsTensor = DecoderLstm.Forward(decoderInputs);
#if false
            Console.WriteLine("данные перед передачей в модель");
            HostReturner.PrintTensor(decoderOutputsTensor);
#endif
            // === ШАГ 4: ПРОЕКЦИЯ В СЛОВАРЬ ===
            Tensor logits = FullyConnectedOut.Forward(decoderOutputsTensor); // [Batch, SeqLen, VocabSize]
#if false
            Console.WriteLine("данные прямого прохода encoder-decoder:");
            HostReturner.PrintTensor(logits);
#endif
            return logits;
        }
    }
}
