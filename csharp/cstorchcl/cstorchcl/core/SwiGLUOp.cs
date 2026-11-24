using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    /// <summary>
    /// Узел графа автодифференцирования для SwiGLU.
    /// </summary>
    public class SwiGLUOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly int _intermediateDim;
        private readonly string _layerId;

        public SwiGLUOp(Tensor x, Tensor weightGateUp, Tensor output, int intermediateDim, NnClGlob nnClGlob, string layerId)
        {
            Inputs = new[] { x, weightGateUp };
            Output = output;
            _intermediateDim = intermediateDim;
            _nnClGlob = nnClGlob;
            _layerId = layerId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            var (gradX, gradW) = _nnClGlob.SwiGLUGateUpBackward(
                gradOutput, Inputs[0], Inputs[1], _intermediateDim, _layerId);

            return new[] { gradX, gradW };
        }
    }
}
