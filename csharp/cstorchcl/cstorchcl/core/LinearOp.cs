using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;

public class LinearOp : Op
{
    private readonly NnClGlob _nnClGlob;

    public LinearOp(Tensor x, Tensor w, Tensor bias, Tensor output, NnClGlob nnClGlob)
    {
        Inputs = new[] { x, w, bias };
        Output = output;
        _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
    }

    public override Tensor[] Backward(Tensor gradOutput, Context context)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

        Tensor x = Inputs[0];
        Tensor w = Inputs[1];
        Tensor bias = Inputs[2];

        Tensor gradX = null;
        Tensor gradW = null;
        Tensor gradBias = null;

        // 1. gradX = gradOutput * W^T (transB: true)
        if (x.RequiresGrad)
        {
            gradX = _nnClGlob.BmmPermute(gradOutput, w, transA: false, transB: true);
        }

        // 2. gradW = Sum_Batch(X^T * gradOutput) (transA: true)
        if (w.RequiresGrad)
        {
            using Tensor batchGradW = _nnClGlob.BmmPermute(x, gradOutput, transA: true, transB: false);
            gradW = _nnClGlob.Sum(batchGradW, axis: 0, keepDims: true);
            gradW.Creator = null;
        }

        // 3. gradBias
        if (bias.RequiresGrad)
        {
            gradBias = _nnClGlob.LinearBiasBackward(gradOutput, bias.Shape);
        }

        return new[] { gradX, gradW, gradBias };
    }

}
