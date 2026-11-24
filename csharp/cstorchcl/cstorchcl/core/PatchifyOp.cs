using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

public class PatchifyOp : Op
{
    private readonly NnClGlob _nnClGlob;
    private readonly int _patchSize;
    private readonly string _opId;

    public PatchifyOp(Tensor img, Tensor output, int patchSize, NnClGlob nnClGlob, string opId)
    {
        Inputs = new[] { img };
        Output = output;
        _patchSize = patchSize;
        _nnClGlob = nnClGlob;
        _opId = opId;
    }

    public override Tensor[] Backward(Tensor gradOutput, Context context)
    {
        Tensor gradImg = _nnClGlob.PatchifyBackward(gradOutput, Inputs[0].Shape, _patchSize, _opId);
        return new[] { gradImg };
    }
}