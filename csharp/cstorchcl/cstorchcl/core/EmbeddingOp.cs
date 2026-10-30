using System;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для слоя выборки эмбеддингов (Embedding).
    /// </summary>
    public class EmbeddingOp : Op
    {
        private readonly NOpenCL.Buffer _xTokensBuf;
        private readonly int _totalTokens;
        private readonly NnClGlob _nnClGlob;

        /// <summary>
        /// Создает узел операции Embedding в графе вычислений.
        /// </summary>
        public EmbeddingOp(NOpenCL.Buffer xTokensBuf, Tensor weight, Tensor output, int totalTokens, NnClGlob nnClGlob)
        {
            // У операции один дифференцируемый вход — матрица весов (Weight)
            Inputs = new[] { weight };
            Output = output;

            _xTokensBuf = xTokensBuf ?? throw new ArgumentNullException(nameof(xTokensBuf));
            _totalTokens = totalTokens;
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиент функции ошибки по матрице весов Embedding.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

            Tensor weight = Inputs[0];
            Tensor gradWeight = null;

            if (weight.RequiresGrad)
            {
                // Запускаем Си-ядро обратного прохода с атомарным сложением на GPU
                gradWeight = _nnClGlob.EmbeddingBackward(gradOutput, _xTokensBuf, weight, _totalTokens);
            }

            // Возвращаем массив градиентов по количеству входов в Inputs (в данном случае один элемент)
            return new[] { gradWeight };
        }
    }
}
