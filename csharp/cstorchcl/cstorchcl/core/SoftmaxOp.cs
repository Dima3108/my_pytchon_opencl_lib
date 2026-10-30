using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для функции активации Softmax.
    /// </summary>
    public class SoftmaxOp : Op
    {
        private readonly NnClGlob _nnClGlob;

        /// <summary>
        /// Создает узел операции Softmax в графе вычислений.
        /// </summary>
        /// <param name="input">Входной тензор X.</param>
        /// <param name="output">Выходной тензор Y (результат прямого прохода).</param>
        /// <param name="nnClGlob">Диспетчер глобальных ядер.</param>
        public SoftmaxOp(Tensor input, Tensor output, NnClGlob nnClGlob)
        {
            Inputs = new[] { input };
            Output = output;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиент функции Softmax по входу X.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput), "Градиент, пришедший сверху, не может быть null.");

            // Для расчета градиента Softmax нам нужен сохраненный выход прямого прохода (Output)
            // Формула требует поэлементных операций над пришедшим градиентом и вероятностями Softmax
            Tensor gradInput = _nnClGlob.SoftmaxBackward(gradOutput, Output);

            // Возвращаем массив градиентов по количеству входов. У Softmax один вход X, поэтому массив из 1 элемента.
            return new[] { gradInput };
        }
    }
}
