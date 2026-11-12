using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class SumOp : Op
    {
        private readonly int _axis;
        private readonly NnClGlob _nnClGlob;

        public SumOp(Tensor input, Tensor output, int axis, NnClGlob nnClGlob)
        {
            Inputs = new[] { input };
            Output = output;
            _axis = axis;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

            Tensor input = Inputs[0];
            Tensor gradInput = _nnClGlob.SumBackward(gradOutput, input.Shape, _axis);

            return new[] { gradInput };
        }
    }

}
