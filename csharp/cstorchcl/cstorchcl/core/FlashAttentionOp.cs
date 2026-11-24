using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class FlashAttentionOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly int _numHeads;
        private readonly bool _isCausal;
        private readonly string _layerId;

        public FlashAttentionOp(Tensor q, Tensor k, Tensor v, Tensor output, int numHeads, bool isCausal, NnClGlob nnClGlob, string layerId)
        {
            Inputs = new[] { q, k, v };
            Output = output;
            _numHeads = numHeads;
            _isCausal = isCausal;
            _nnClGlob = nnClGlob;
            _layerId = layerId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            var (gradQ, gradK, gradV) = _nnClGlob.FlashAttentionBackward(
                gradOutput, Inputs[0], Inputs[1], Inputs[2], Output, _numHeads, _isCausal, _layerId);

            return new[] { gradQ, gradK, gradV };
        }
    }
}
