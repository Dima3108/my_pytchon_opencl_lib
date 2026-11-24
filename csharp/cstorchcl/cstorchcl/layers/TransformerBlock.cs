using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.layers
{
    public class TransformerBlock : NeuralNetwork
    {
        public Tensor Norm1Gamma { get; private set; }
        public MultiHeadAttention Attention { get; private set; }
        public Tensor Norm2Gamma { get; private set; }
        public SwiGLU FeedForward { get; private set; }

        public unsafe TransformerBlock(
            ComputeContext computeContext,
            NnClGlob nnClGlob,
            int dModel,
            int numHeads,
            int intermediateDim,
            bool isCausal,
            string layerId)
            : base(computeContext, nnClGlob, layerId)
        {
            // 1. Веса первой нормализации RMSNorm (инициализируются единицами, как в PyTorch)
            Norm1Gamma = RegisterParameter(new[] { dModel }, requiresGrad: true, tenosr_name: "norm1.weight");
            Norm2Gamma = RegisterParameter(new[] { dModel }, requiresGrad: true, tenosr_name: "norm2_.weight");

            float[] ones = new float[dModel];
            Array.Fill(ones, 1.0f);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            fixed (float* pOnes = ones)
            {
                long bytes = (long)dModel * sizeof(float);
                queue.EnqueueWriteBuffer(Norm1Gamma.DataBuffer, true, 0, bytes, (IntPtr)pOnes);
                queue.EnqueueWriteBuffer(Norm2Gamma.DataBuffer, true, 0, bytes, (IntPtr)pOnes);
            }
            queue.Finish();
            //Norm1Gamma.InitGradient(queue.Context);
            //Norm2Gamma.InitGradient(queue.Context);
            // 2. Слой внимания (FlashAttention)
            Attention = new MultiHeadAttention(computeContext, nnClGlob, dModel, numHeads, isCausal, $"{layerId}.attn");

            // 3. Слой MLP (SwiGLU)
            FeedForward = new SwiGLU(computeContext, nnClGlob, dModel, intermediateDim, $"{layerId}.ffn");
        }

        public new IEnumerable<Tensor> GetParameters()
        {
            // Возвращаем norm1.weight, norm2.weight + все веса Attention + все веса SwiGLU
            return base.GetParameters()
                .Concat(Attention.GetParameters())
                .Concat(FeedForward.GetParameters());
        }

        public Tensor Forward(Tensor x)
        {
            // Ветка 1: x + Attention(RMSNorm(x))
            Tensor norm1 = NnClGlob.RmsNorm(x, Norm1Gamma,LAYER_ID+".rs1");
            //HostReturner.PrintTensor(norm1, "norm1", true);
            Tensor attnOut = Attention.Forward(norm1);
            //HostReturner.PrintTensor(attnOut, "attnOut", true);
            Tensor h = NnClGlob.Add(x, attnOut, $"{LAYER_ID}.res1");
            //HostReturner.PrintTensor(h, "h", true);
            // Ветка 2: h + SwiGLU(RMSNorm(h))
            Tensor norm2 = NnClGlob.RmsNorm(h, Norm2Gamma,LAYER_ID+".rs2");
            //HostReturner.PrintTensor(norm2, "norm2", true);
            Tensor ffnOut = FeedForward.Forward(norm2);
            //HostReturner.PrintTensor(ffnOut, "ffnOut", true);
            Tensor outTensor = NnClGlob.Add(h, ffnOut, $"{LAYER_ID}.res2");
            //HostReturner.PrintTensor(outTensor, "outTensor", true);
            return outTensor;
        }
    }
}
