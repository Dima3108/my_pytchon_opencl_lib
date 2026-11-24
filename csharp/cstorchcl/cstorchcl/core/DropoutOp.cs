using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    public class DropoutOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        private readonly float _scale;
        private readonly string _layerId;

        public DropoutOp(Tensor x, Tensor y, float scale, NnClGlob nnClGlob, string layerId)
        {
            Inputs = new[] { x };
            Output = y;
            _scale = scale;
            _nnClGlob = nnClGlob;
            _layerId = layerId;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            using LayerBuffers buffers = GradientStorage.Pop();
            if (!buffers.layerBuffers.TryGetValue($"{_layerId}_mask", out NOpenCL.Buffer? maskBuf))
                throw new InvalidOperationException($"Маска '{_layerId}_mask' не найдена для обратного прохода Dropout.");

            Tensor gradX = _nnClGlob.DropoutBackward(gradOutput, maskBuf, _scale, _layerId);
            return new[] { gradX };
        }
    }
}
