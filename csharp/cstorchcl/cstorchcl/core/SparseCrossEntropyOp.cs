using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class SparseCrossEntropyOp : Op
    {
        private readonly NOpenCL.Buffer _targetsBuf;
        private readonly NnClGlob _nnClGlob;

        public SparseCrossEntropyOp(Tensor logits, NOpenCL.Buffer targetsBuf, Tensor lossOutput, NnClGlob nnClGlob)
        {
            Inputs = new Tensor[] { logits };
            Output = lossOutput;
            _targetsBuf = targetsBuf;
            _nnClGlob = nnClGlob;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            Tensor gradLogits = _nnClGlob.SparseCrossEntropyBackward(gradOutput, Inputs[0], _targetsBuf);
            return new Tensor[] { gradLogits };
        }
    }

}
