using cstorchcl.compiler;
using NOpenCL;
using System;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для усечения тензора логитов по временной оси.
    /// </summary>
    public class SliceOp : Op
    {
        private readonly NnClGlob _nnClGlob;
        /// <summary>
        /// Создает узел операции Slice в вычислительном графе.
        /// </summary>
        /// <param name="input">Исходный широкий тензор логитов [Batch, SeqLen, VocabSize].</param>
        /// <param name="output">Усеченный выходной тензор [Batch, SeqLen - 1, VocabSize].</param>
        public SliceOp(Tensor input, Tensor output, NnClGlob nnClGlob)
        {
            Inputs = new Tensor[] { input };
            Output = output;
            _nnClGlob = nnClGlob;
        }

        /// <summary>
        /// Выполняет обратный проход: расширяет пришедший градиент до оригинальной формы, 
        /// дополняя отсутствующие временные шаги нулями во VRAM.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput), "Градиент, пришедший сверху в SliceOp, не может быть null.");

            Tensor originalLogits = Inputs[0];

            if (originalLogits.RequiresGrad)
            {
                // Лениво инициализируем буфер градиента для исходных логитов
                originalLogits.InitGradient(context);

                // Физически расширяем усеченный градиент и переносим данные во VRAM родителя
                _nnClGlob.PadSlicedGradient(gradOutput, originalLogits.Grad);
            }

            // Возвращаем null, так как мы записали градиент напрямую в .Grad родительского тензора,
            // минуя промежуточные вычисления и циклическое суммирование на CPU хосте.
            return new Tensor[] { null };
        }
    }
}
