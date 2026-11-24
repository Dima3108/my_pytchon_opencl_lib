using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class AddOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly string _opId;

        public AddOp(Tensor a, Tensor b, Tensor output, NnClGlob nnClGlob, string opId)
        {
            Inputs = new[] { a, b };
            Output = output;
            _nnClGlob = nnClGlob;
            _opId = opId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            Tensor gradA = _nnClGlob.CloneTensor(gradOutput, $"{_opId}_grad_a");
            Tensor gradB = _nnClGlob.CloneTensor(gradOutput, $"{_opId}_grad_b");
            return new[] { gradA, gradB };
        }
    }

}
