using cstorchcl.compiler;
using NOpenCL;
using System;

namespace cstorchcl.core
{
    /// <summary>
    /// Узел графа автодифференцирования для операции слияния тензоров по последней оси.
    /// </summary>
    public class ConcatOp : Op
    {
        private readonly int _hiddenDim;

        public ConcatOp(Tensor left, Tensor right, Tensor output, int hiddenDim)
        {
            // Регистрируем оба входящих тензора (левый эмбеддинг и правый выход энкодера)
            Inputs = new Tensor[] { left, right };
            Output = output;
            _hiddenDim = hiddenDim;
        }

        /// <summary>
        /// Принимает градиент dL/dDest и распределяет его в .Grad буферы входящих тензоров.
        /// </summary>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput), "Градиент, пришедший сверху, не может быть null.");

            Tensor left = Inputs[0];
            Tensor right = Inputs[1];

            // Лениво выделяем/готовим буферы градиентов для родителей
            left.InitGradient(context);
            right.InitGradient(context);

            // Физически разрезаем пришедший градиент во VRAM
            NnClGlob.SplitConcatGradients(gradOutput, left.Grad, right.Grad, _hiddenDim);

            // Возвращаем null-массив для движка, так как мы УЖЕ записали результаты 
            // напрямую в .Grad поля родительских тензоров левой и правой ветки.
            // Это избавляет хост от лишнего циклического суммирования.
            return new Tensor[] { null, null };
        }
    }
}
