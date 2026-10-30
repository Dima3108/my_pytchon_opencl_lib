using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для пакетного матричного умножения (BMM).
    /// </summary>
    public class BmmOp : Op
    {
        private readonly NnClGlob _nnClGlob;

        /// <summary>
        /// Создает узел операции BMM в графе вычислений.
        /// </summary>
        public BmmOp(Tensor a, Tensor b, Tensor output, NnClGlob nnClGlob)
        {
            Inputs = new[] { a, b };
            Output = output;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиенты по матрицам А и В.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

            Tensor a = Inputs[0];
            Tensor b = Inputs[1];

            Tensor gradA = null;
            Tensor gradB = null;

            // Массив индексов осей для транспонирования двух последних размерностей 3D-тензора: [Batch, M, N] -> [Batch, N, M]
            int[] transposeAxes = { 0, 2, 1 };

            // 1. Вычисляем gradA = gradOutput * B.T
            if (a.RequiresGrad)
            {
                // Транспонируем матрицу B по двум последним осям
                using Tensor bT = _nnClGlob.Permute(b, transposeAxes);

                // Перемножаем пришедший сверху градиент на транспонированную матрицу B
                gradA = _nnClGlob.Bmm(gradOutput, bT);

                // Сбрасываем флаг автоградиента для промежуточного тензора, чтобы он не забивал граф
                gradA.Creator = null;
            }

            // 2. Вычисляем gradB = A.T * gradOutput
            if (b.RequiresGrad)
            {
                // Транспонируем матрицу A по двум последним осям
                using Tensor aT = _nnClGlob.Permute(a, transposeAxes);

                // Перемножаем транспонированную матрицу A на пришедший сверху градиент
                gradB = _nnClGlob.Bmm(aT, gradOutput);

                // Сбрасываем флаг автоградиента для промежуточного тензора
                gradB.Creator = null;
            }

            // Возвращаем вычисленные градиенты в соответствии с порядком во входах Inputs
            return new[] { gradA, gradB };
        }
    }
}
