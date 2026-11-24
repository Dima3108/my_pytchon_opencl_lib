using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    public class RMSNormOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly NOpenCL.Buffer _invRmsCache;
        private readonly string _layerId;

        public RMSNormOp(Tensor x, Tensor gamma, Tensor output, NOpenCL.Buffer invRmsCache, NnClGlob nnClGlob, string layerId)
        {
            Inputs = new[] { x, gamma };
            Output = output;
            _invRmsCache = invRmsCache ?? throw new ArgumentNullException(nameof(invRmsCache));
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
            _layerId = layerId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

            Tensor x = Inputs[0];
            Tensor gamma = Inputs[1];

            // Передаем кэш напрямую, без GradientStorage.Pop()
            var (gradX, gradGamma) = _nnClGlob.RmsNormBackward(gradOutput, x, gamma, _invRmsCache, _layerId);

            return new[] { gradX, gradGamma };
        }
    }

}
