using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;

public class TokenShiftOp : Op
{
    private readonly NnClGlob _glob;
    private readonly string _id;
    public TokenShiftOp(Tensor x, Tensor mu, Tensor y, NnClGlob glob, string id)
    { Inputs = new[] { x, mu }; Output = y; _glob = glob; _id = id; }

    public override Tensor[] Backward(Tensor gradOut, Context ctx)
    {
        var (gx, gmu) = _glob.TokenShiftBackward(gradOut, Inputs[0], Inputs[1], _id);
        return new[] { gx, gmu };
    }
}

public class RwkvUnaryOp : Op
{
    private readonly NnClGlob _glob;
    private readonly int _mode;
    private readonly string _id;
    public RwkvUnaryOp(Tensor x, Tensor y, int mode, NnClGlob glob, string id)
    { Inputs = new[] { x }; Output = y; _mode = mode; _glob = glob; _id = id; }

    public override Tensor[] Backward(Tensor gradOut, Context ctx)
        => new[] { _glob.RwkvUnaryBackward(gradOut, Inputs[0], Output, _mode, _id) };
}

public class MulOp : Op
{
    private readonly NnClGlob _glob;
    private readonly string _id;
    public MulOp(Tensor a, Tensor b, Tensor c, NnClGlob glob, string id)
    { Inputs = new[] { a, b }; Output = c; _glob = glob; _id = id; }

    public override Tensor[] Backward(Tensor gradOut, Context ctx)
    {
        var (ga, gb) = _glob.MulBackward(gradOut, Inputs[0], Inputs[1], _id);
        return new[] { ga, gb };
    }
}