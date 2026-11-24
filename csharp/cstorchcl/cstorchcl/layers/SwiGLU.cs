using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.layers
{
    public class SwiGLU : NeuralNetwork
    {
        /// <summary>
        /// Объединённая матрица весов [1, D_model, 2 * IntermediateDim]
        /// </summary>
        public Tensor WeightGateUp { get; private set; }

        /// <summary>
        /// Выходная проекция [Batch, SeqLen, IntermediateDim] -> [Batch, SeqLen, D_model]
        /// </summary>
        public Linear WDown { get; private set; }

        private readonly int _dModel;
        private readonly int _intermediateDim;

        public unsafe SwiGLU(ComputeContext computeContext, NnClGlob nnClGlob, int dModel, int intermediateDim, string layerId)
            : base(computeContext, nnClGlob, layerId)
        {
            _dModel = dModel;
            _intermediateDim = intermediateDim;

            WeightGateUp = RegisterParameter(
                new[] { 1, dModel, 2 * intermediateDim },
                requiresGrad: true,
                tenosr_name: nameof(WeightGateUp));

            // Инициализация Ксавье
            var rand = new Random();
            float bound = (float)(Math.Sqrt(6.0) / Math.Sqrt(dModel + intermediateDim));
            float[] hostW = new float[WeightGateUp.Size];
            for (int i = 0; i < hostW.Length; i++)
                hostW[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * bound;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            fixed (float* pW = hostW)
            {
                queue.EnqueueWriteBuffer(WeightGateUp.DataBuffer, true, 0, (long)WeightGateUp.Size * sizeof(float), (IntPtr)pW);
            }
            queue.Finish();

            WDown = new Linear(computeContext, nnClGlob, intermediateDim, dModel, $"{layerId}.w_down");
        }

        public new IEnumerable<Tensor> GetParameters()
        {
            return base.GetParameters().Concat(WDown.GetParameters());
        }

        public Tensor Forward(Tensor x)
        {
            // 1. Объединенный проход Gate + Up + SiLU
            Tensor gated = NnClGlob.SwiGLUGateUp(x, WeightGateUp, _intermediateDim, $"{LAYER_ID}.gate_up");
            // 2. Проекция обратно в D_model через ваш готовый Linear
            return WDown.Forward(gated);
        }
    }
}
