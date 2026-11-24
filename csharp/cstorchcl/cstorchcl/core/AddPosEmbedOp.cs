using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class AddPosEmbedOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly string _opId;

        public AddPosEmbedOp(Tensor x, Tensor posEmbed, Tensor output, NnClGlob nnClGlob, string opId)
        {
            Inputs = new[] { x, posEmbed };
            Output = output;
            _nnClGlob = nnClGlob;
            _opId = opId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            var (gradX, gradPos) = _nnClGlob.AddPosEmbedBackward(gradOutput, Inputs[1], _opId);
            return new[] { gradX, gradPos };
        }
    }
}
