#if DEBUG
using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    public class EluOp : Op
    {
        private readonly float _alpha;
        private readonly NnClGlob _nnClGlob;

        public EluOp(Tensor input, Tensor output, float alpha, NnClGlob nnClGlob)
        {
            Inputs = new[] { input };
            Output = output;
            _alpha = alpha;
            _nnClGlob = nnClGlob;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            // Передаем в Backward сохраненный на этапе Forward выход модели (Output)
            Tensor gradInput = _nnClGlob.EluBackward(gradOutput, Output, _alpha);
            return new[] { gradInput };
        }
    }
}
#endif
