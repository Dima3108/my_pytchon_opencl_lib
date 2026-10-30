using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для нормализации RMSNorm.
    /// </summary>
    public class RMSNormOp : Op
    {
        private readonly NnClGlob _nnClGlob;

        /// <summary>
        /// Создает узел операции RMSNorm в графе вычислений.
        /// </summary>
        /// <param name="x">Входной тензор скрытых состояний X.</param>
        /// <param name="gamma">Входной тензор обучаемых весов Gamma.</param>
        /// <param name="output">Выходной тензор Y.</param>
        /// <param name="nnClGlob">Диспетчер глобальных ядер.</param>
        public RMSNormOp(Tensor x, Tensor gamma, Tensor output, NnClGlob nnClGlob)
        {
            // У операции RMSNorm два входа: данные и веса слоя
            Inputs = new[] { x, gamma };
            Output = output;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиенты функции RMSNorm по входу X и по весам Gamma.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            Tensor x = Inputs[0];
            Tensor gamma = Inputs[1];

            // Извлекаем кэш инвертированного RMS, сохраненный на этапе Forward прохода.
            // Принцип LIFO гарантирует, что мы получим кэш именно этого слоя.
            using LayerBuffers buffers = GradientStorage.Pop();

            if (!buffers.layerBuffers.TryGetValue("inv_rms_cache", out NOpenCL.Buffer? invRmsCache))
            {
                throw new InvalidOperationException("Кэш 'inv_rms_cache' не найден для обратного прохода RMSNorm.");
            }

            // Вызываем выполнение Си-ядра обратного прохода и редукцию весов
            var (gradX, gradGamma) = _nnClGlob.RmsNormBackward(gradOutput, x, gamma, invRmsCache);

            // Возвращаем градиенты в порядке следования входов в массиве Inputs:
            // Индекс 0: Градиент по входу X
            // Индекс 1: Градиент по весам Gamma
            return new[] { gradX, gradGamma };
        }
    }
}
