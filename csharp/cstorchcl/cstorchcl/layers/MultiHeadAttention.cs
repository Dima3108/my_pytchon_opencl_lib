using cstorchcl.compiler;
using cstorchcl.core;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.layers
{
    public class MultiHeadAttention : NeuralNetwork
    {
        public Linear Wq { get; private set; }
        public Linear Wk { get; private set; }
        public Linear Wv { get; private set; }
        public Linear Wo { get; private set; }

        private readonly int _numHeads;
        private readonly bool _isCausal;

        public MultiHeadAttention(ComputeContext computeContext, NnClGlob nnClGlob, int dModel, int numHeads, bool isCausal, string layerId)
            : base(computeContext, nnClGlob, layerId)
        {
            if (dModel % numHeads != 0)
                throw new System.ArgumentException("dModel должен делиться на numHeads без остатка.");

            _numHeads = numHeads;
            _isCausal = isCausal;

            Wq = new Linear(computeContext, nnClGlob, dModel, dModel, $"{layerId}.wq");
            Wk = new Linear(computeContext, nnClGlob, dModel, dModel, $"{layerId}.wk");
            Wv = new Linear(computeContext, nnClGlob, dModel, dModel, $"{layerId}.wv");
            Wo = new Linear(computeContext, nnClGlob, dModel, dModel, $"{layerId}.wo");
        }

        public new IEnumerable<Tensor> GetParameters()
        {
            return Wq.GetParameters()
                .Concat(Wk.GetParameters())
                .Concat(Wv.GetParameters())
                .Concat(Wo.GetParameters());
        }

        public Tensor Forward(Tensor x)
        {
            Tensor q = Wq.Forward(x);
            //HostReturner.PrintTensor(q, $"linear_{q.TensorID}", true);
            Tensor k = Wk.Forward(x);
            Tensor v = Wv.Forward(x);

            Tensor attnOut = NnClGlob.FlashAttention(q, k, v, _numHeads, _isCausal, $"{LAYER_ID}.attn");
            return Wo.Forward(attnOut);
        }
    }
}
