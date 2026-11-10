// Класс узла операции графа Autograd
using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;

public class CrossEntropyLossOp : Op
{
    private readonly NnClGlob _nnClGlob;
    public CrossEntropyLossOp(Tensor pred, Tensor target, Tensor output, NnClGlob nnClGlob)
    {
        Inputs = new[] { pred, target };
        Output = output;
        _nnClGlob = nnClGlob;
    }

    public override Tensor[] Backward(Tensor gradOutput, Context context)
    {
        // Вычисляем сопряженный градиент по предсказаниям
        Tensor gradPred = _nnClGlob.CrossEntropyLossBackward(Inputs[0], Inputs[1]);
        return new[] { gradPred, null }; // По One-Hot таргету градиент не собираем
    }
}