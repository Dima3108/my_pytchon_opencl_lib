using System;
using System.Collections.Generic;
using System.Linq;
using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;

namespace cstorchcl.layers
{
    public class DecoderLLM : NeuralNetwork
    {
        public Tensor EmbeddingWeight { get; private set; }
        public List<TransformerBlock> Layers { get; private set; }
        public Tensor FinalNormGamma { get; private set; }
        public Linear LmHead { get; private set; }

        public int VocabSize { get; }
        public int DModel { get; }
        public int NumLayers { get; }

        public unsafe DecoderLLM(
            ComputeContext computeContext,
            NnClGlob nnClGlob,
            int vocabSize,
            int dModel,
            int numLayers,
            int numHeads,
            int intermediateDim,
            string modelId = "decoder_llm")
            : base(computeContext, nnClGlob, modelId)
        {
            VocabSize = vocabSize;
            DModel = dModel;
            NumLayers = numLayers;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            var rand = new Random(42);

            // 1. Эмбеддинги токенов [VocabSize, D_model] (нормальная / равномерная малая инициализация)
            EmbeddingWeight = RegisterParameter(
                new[] { vocabSize, dModel },
                requiresGrad: true,
                tenosr_name: "embed_tokens.weight");

            float embStd = 1.0f / (float)Math.Sqrt(dModel);
            float[] hostEmb = new float[EmbeddingWeight.Size];
            for (int i = 0; i < hostEmb.Length; i++)
            {
                hostEmb[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * embStd;
            }
            fixed (float* pEmb = hostEmb)
            {
                queue.EnqueueWriteBuffer(EmbeddingWeight.DataBuffer, true, 0, (long)EmbeddingWeight.Size * sizeof(float), (IntPtr)pEmb);
            }

            // 2. Стек Decoder-блоков (Pre-RMSNorm + Causal FlashAttention с RoPE + SwiGLU)
            Layers = new List<TransformerBlock>(numLayers);
            for (int i = 0; i < numLayers; i++)
            {
                Layers.Add(new TransformerBlock(
                    computeContext,
                    nnClGlob,
                    dModel: dModel,
                    numHeads: numHeads,
                    intermediateDim: intermediateDim,
                    isCausal: true,
                    layerId: $"{modelId}.layer_{i}"));
            }

            // 3. Финальная RMSNorm перед проекцией в словарь
            FinalNormGamma = RegisterParameter(
                new[] { dModel },
                requiresGrad: true,
                tenosr_name: "final_norm.weight");

            float[] ones = new float[dModel];
            Array.Fill(ones, 1.0f);
            fixed (float* pOnes = ones)
            {
                queue.EnqueueWriteBuffer(FinalNormGamma.DataBuffer, true, 0, (long)dModel * sizeof(float), (IntPtr)pOnes);
            }
            queue.Finish();

            // 4. Выходной линейный слой (LM Head): [Batch, SeqLen, D_model] -> [Batch, SeqLen, VocabSize]
            LmHead = new Linear(computeContext, nnClGlob, dModel, vocabSize, $"{modelId}.lm_head");
        }

        /// <summary>
        /// Собирает все обучаемые параметры модели (совместимо с Adam и SafeTensorsSerializer).
        /// </summary>
        public new IEnumerable<Tensor> GetParameters()
        {
            // EmbeddingWeight и FinalNormGamma зарегистрированы в base
            IEnumerable<Tensor> allParams = base.GetParameters();

            foreach (var layer in Layers)
            {
                allParams = allParams.Concat(layer.GetParameters());
            }

            allParams = allParams.Concat(LmHead.GetParameters());
            return allParams;
        }

        /// <summary>
        /// Прямой проход Decoder-only LLM.
        /// inputTokenIdsBuf: GPU-буфер int[] размера [Batch * SeqLen].
        /// Возвращает логиты формы [Batch, SeqLen, VocabSize].
        /// </summary>
        public Tensor Forward(NOpenCL.Buffer inputTokenIdsBuf, int batchSize, int seqLen)
        {
            int totalTokens = batchSize * seqLen;

            // 1. Получаем 2D-тензор эмбеддингов [batchSize * seqLen, DModel]
            Tensor x = NnClGlob.Embedding(inputTokenIdsBuf, EmbeddingWeight, totalTokens, $"{LAYER_ID}.emb");

            // 2. Меняем логическую форму на 3D [batchSize, seqLen, DModel] за 0 мс (без копирования VRAM)
            x.ReShape(new[] { batchSize, seqLen, DModel });

            // 3. Прогоняем 3D-тензор через стек TransformerBlock
            for (int i = 0; i < Layers.Count; i++)
            {
                x = Layers[i].Forward(x);
            }

            // 4. Финальная нормализация и проекция в словарь [batchSize, seqLen, VocabSize]
            Tensor normOut = NnClGlob.RmsNorm(x, FinalNormGamma, $"{LAYER_ID}.final_norm");
            Tensor logits = LmHead.Forward(normOut);

            return logits;
        }

    }
}
