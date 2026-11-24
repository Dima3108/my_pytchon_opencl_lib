using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.layers
{
  
    using global::cstorchcl.compiler;
    using global::cstorchcl.core;
    using NOpenCL;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    namespace cstorchcl.layers
    {
        public class VisionEncoder : NeuralNetwork
        {
            public Linear PatchProj { get; private set; }
            public Tensor PosEmbed { get; private set; }
            public List<TransformerBlock> Layers { get; private set; }
            public Tensor PostNormGamma { get; private set; }
            public Linear MultimodalProjector { get; private set; }

            public int ImageSize { get; }
            public int PatchSize { get; }
            public int NumPatches { get; }

            public unsafe VisionEncoder(
                ComputeContext computeContext,
                NnClGlob nnClGlob,
                int imageSize = 224,
                int patchSize = 16,
                int inChannels = 3,
                int visionDim = 256,
                int numLayers = 4,
                int numHeads = 4,
                int visionFfnDim = 688,
                int textDModel = 256,
                string modelId = "vision_encoder")
                : base(computeContext, nnClGlob, modelId)
            {
                ImageSize = imageSize;
                PatchSize = patchSize;
                NumPatches = (imageSize / patchSize) * (imageSize / patchSize); // 196 для 224x224 с патчем 16
                int patchDim = inChannels * patchSize * patchSize;              // 768 для RGB 16x16

                // 1. Линейная проекция патчей [Batch, 196, 768] -> [Batch, 196, VisionDim]
                PatchProj = new Linear(computeContext, nnClGlob, patchDim, visionDim, $"{modelId}.patch_proj");

                // 2. Обучаемые 2D-позиционные эмбеддинги [1, 196, VisionDim]
                PosEmbed = RegisterParameter(
                    new[] { 1, NumPatches, visionDim },
                    requiresGrad: true,
                    tenosr_name: "pos_embed.weight");

                CommandQueue queue = DeviceQueueRegistry.GetCurrent();
                var rand = new Random(42);
                float posStd = 0.02f;
                float[] hostPos = new float[PosEmbed.Size];
                for (int i = 0; i < hostPos.Length; i++)
                    hostPos[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * posStd;

                fixed (float* pPos = hostPos)
                {
                    queue.EnqueueWriteBuffer(PosEmbed.DataBuffer, true, 0, (long)PosEmbed.Size * sizeof(float), (IntPtr)pPos);
                }

                // 3. Стек Vision Transformer блоков (ВАЖНО: isCausal = false, патчи видят всю картинку целиком)
                Layers = new List<TransformerBlock>(numLayers);
                for (int i = 0; i < numLayers; i++)
                {
                    Layers.Add(new TransformerBlock(
                        computeContext,
                        nnClGlob,
                        dModel: visionDim,
                        numHeads: numHeads,
                        intermediateDim: visionFfnDim,
                        isCausal: false,
                        layerId: $"{modelId}.vit_layer_{i}"));
                }

                // 4. Финальная нормализация и мультимодальный проектор в пространство текстовой LLM
                PostNormGamma = RegisterParameter(new[] { visionDim }, requiresGrad: true, tenosr_name: "post_norm.weight");
                float[] ones = new float[visionDim];
                Array.Fill(ones, 1.0f);
                fixed (float* pOnes = ones)
                {
                    queue.EnqueueWriteBuffer(PostNormGamma.DataBuffer, true, 0, (long)visionDim * sizeof(float), (IntPtr)pOnes);
                }
                queue.Finish();

                MultimodalProjector = new Linear(computeContext, nnClGlob, visionDim, textDModel, $"{modelId}.mm_projector");
            }

            public new IEnumerable<Tensor> GetParameters()
            {
                // PosEmbed и PostNormGamma зарегистрированы в base
                IEnumerable<Tensor> allParams = base.GetParameters()
                    .Concat(PatchProj.GetParameters());

                foreach (var layer in Layers)
                {
                    allParams = allParams.Concat(layer.GetParameters());
                }

                allParams = allParams.Concat(MultimodalProjector.GetParameters());
                return allParams;
            }

            /// <summary>
            /// Прямой проход: Изображение [B, 3, H, W] -> Визуальные токены [B, NumPatches, TextDModel].
            /// </summary>
            public Tensor Forward(Tensor images)
            {
                // 1. Разрезаем картинку на патчи [B, 196, 768]
                Tensor patches = NnClGlob.Patchify(images, PatchSize, $"{LAYER_ID}.patchify");

                // 2. Проецируем патчи в эмбеддинги [B, 196, VisionDim]
                Tensor x = PatchProj.Forward(patches);

                // 3. Прибавляем 2D-позиционные эмбеддинги
                x = NnClGlob.AddPosEmbed(x, PosEmbed, $"{LAYER_ID}.add_pos");

                // 4. Прогоняем через двунаправленные ViT-блоки
                for (int i = 0; i < Layers.Count; i++)
                {
                    x = Layers[i].Forward(x);
                }

                // 5. Финальная нормализация и проекция в пространство текстовых токенов LLM
                Tensor normed = NnClGlob.RmsNorm(x, PostNormGamma, $"{LAYER_ID}.post_norm");
                return MultimodalProjector.Forward(normed);
            }
        }
    }

}
