using cstorchcl.compiler;
using NOpenCL;

namespace cstorchcl.core
{
    public class WkvOp : Op
    {
        private readonly NnClGlob _glob;
        private readonly NOpenCL.Buffer _stateAa;
        private readonly NOpenCL.Buffer _stateBb;
        private readonly NOpenCL.Buffer _statePp;
        private readonly string _layerId;

        public WkvOp(Tensor w, Tensor u, Tensor k, Tensor v, Tensor y,
                     NOpenCL.Buffer stateAa, NOpenCL.Buffer stateBb, NOpenCL.Buffer statePp,
                     NnClGlob glob, string layerId)
        {
            Inputs = new[] { w, u, k, v };
            Output = y;
            _stateAa = stateAa;
            _stateBb = stateBb;
            _statePp = statePp;
            _glob = glob;
            _layerId = layerId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            var (gw, gu, gk, gv) = _glob.WkvBackward(
                Inputs[0], Inputs[1], Inputs[2], Inputs[3], Output, gradOutput,
                _stateAa, _stateBb, _statePp, _layerId
            );
            return new[] { gw, gu, gk, gv };
        }
    }
}
