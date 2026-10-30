using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    public class MseLossOp : Op
    {
        private readonly NnClGlob _nnClGlob;

        public MseLossOp(Tensor pred, Tensor target, Tensor output, NnClGlob nnClGlob)
        {
            Inputs = new[] { pred, target };
            Output = output;
            _nnClGlob = nnClGlob;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            // Нам пришел gradOutput скаляра Loss (он равен 1.0f). Считаем внутренний градиент по предсказаниям.
            Tensor gradPred = _nnClGlob.MseLossBackward(Inputs[0], Inputs[1]);
            return new[] { gradPred, null }; // По таргету Y градиент не нужен (null)
        }
    }
}
