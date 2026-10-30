using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для каузальной одномерной Depthwise свёртки.
    /// </summary>
    public class CausalConv1dOp : Op
    {
        private readonly NnClGlob _nnClGlob;

        /// <summary>
        /// Создает узел операции CausalConv1d в графе вычислений.
        /// </summary>
        public CausalConv1dOp(Tensor x, Tensor weight, Tensor bias, Tensor output, NnClGlob nnClGlob)
        {
            // Три входа: данные X, матрица фильтров Weight, вектор смещений Bias
            Inputs = new[] { x, weight, bias };
            Output = output;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиенты по входу X, весам Weight и смещению Bias за один проход.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

            Tensor x = Inputs[0];
            Tensor weight = Inputs[1];
            Tensor bias = Inputs[2];

            // Вызываем объединенный метод вычисления градиентов свёртки
            var (gradX, gradWeight, gradBias) = _nnClGlob.CausalConv1dBackward(gradOutput, x, weight);

            // Возвращаем массив градиентов строго в порядке позиций элементов в массиве Inputs
            return new[] { gradX, gradWeight, gradBias };
        }
    }
}
