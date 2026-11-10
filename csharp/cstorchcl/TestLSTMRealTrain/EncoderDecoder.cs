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

        private readonly int _hiddenDim;

        public EncoderDecoderNetwork(ComputeContext computeContext, NnClGlob nnClGlob, int hiddenDim, int batchSize)
            : base(computeContext, nnClGlob)
        {
            _hiddenDim = hiddenDim;

            // 1. Энкодер: принимает эмбеддинг токена -> выдает hidden_dim
            EncoderLstm = new LSTM(computeContext, nnClGlob, inFeatures: _hiddenDim, hiddenSize: _hiddenDim);

            // 2. Декодер: принимает конкатенацию (embedding + encoder_output) -> размерность 2 * hidden_dim
            DecoderLstm = new LSTM(computeContext, nnClGlob, inFeatures: _hiddenDim * 2, hiddenSize: _hiddenDim);

            // 3. Финальная проекция в размерность словаря
            FullyConnectedOut = new Linear(computeContext, nnClGlob, inFeatures: _hiddenDim, outFeatures: 6565); // VocabSize = 6565
        }

        /// <summary>
        /// Скорректированный Forward Pass без явного слоя Embedding и без операции Concat.
        /// </summary>
        public Tensor Forward(
            NOpenCL.Buffer encoderInputsBuf,
            NOpenCL.Buffer decoderTargetsBuf,
            Tensor embWeightsEnc, // Явно принимаем тензоры весов эмбеддингов
            Tensor embWeightsDec,
            int seqLen,
            int batchSize)
        {
            // === ШАГ 1: ЭНКОДЕР ===
            // Вызываем эмбеддинг через глобальный объект clGlob
            Tensor embeddedEncoder = NnClGlob.Embedding(encoderInputsBuf, embWeightsEnc, batchSize * seqLen);
            //embeddedEncoder.Shape = new int { batchSize, seqLen, _hiddenDim };
            embeddedEncoder.ReShape(new int []{ batchSize, seqLen, _hiddenDim });
            // Прогон через LSTM Энкодера
            Tensor encoderOutputs = EncoderLstm.Forward(embeddedEncoder); // [Batch, SeqLen, HiddenDim]

            // === ШАГ 2: ДЕКОДЕР ===
            Tensor embeddedDecoder = NnClGlob.Embedding(decoderTargetsBuf, embWeightsDec, batchSize * seqLen);
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
