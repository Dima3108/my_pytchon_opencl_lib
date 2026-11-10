using cstorchcl.compiler;
using cstorchcl.core;


using NOpenCL;


/// <summary>
/// Класс операции графа автодифференцирования для функции Cat.
/// </summary>
public class CatOp : Op
{
    private readonly Tensor _x1;
    private readonly Tensor _x2;
    private readonly NnClGlob _nnClGlob;

    public CatOp(Tensor x1, Tensor x2, Tensor output, NnClGlob nnClGlob)
    {
        Inputs = new[] { x1, x2 };
        Output = output;
        _x1 = x1;
        _x2 = x2;
        _nnClGlob = nnClGlob;
    }

    public override Tensor[] Backward(Tensor gradOutput, Context context)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput), "Градиент не может быть null.");

        var (gradInput1, gradInput2) = _nnClGlob.CatBackward(gradOutput, _x1, _x2);

        return new[] { gradInput1, gradInput2 };
    }
}