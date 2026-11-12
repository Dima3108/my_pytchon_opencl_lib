using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace cstorchcl.core
{
    /// <summary>
    /// Операция графа автодифференцирования для монолитного слоя LSTM.
    /// Хранит ссылки на входные тензоры и контекст промежуточных гейтов (Workspace) во VRAM.
    /// </summary>
    public class LstmOp : Op, IDisposable
    {
        private readonly NnClGlob _nnClGlob;
        private NOpenCL.Buffer _workspaceBuf;
        private bool _isDisposed = false;

        /// <summary>
        /// Создает узел операции LSTM в вычислительном графе.
        /// </summary>
        /// <param name="x">Входной тензор данных [Batch, SeqLen, InFeatures].</param>
        /// <param name="weightW">Матрица весов входа [InFeatures, 4 * HiddenSize].</param>
        /// <param name="weightU">Матрица весов скрытого состояния [HiddenSize, 4 * HiddenSize].</param>
        /// <param name="bias">Вектор смещений гейтов [4 * HiddenSize].</param>
        /// <param name="output">Выходной тензор скрытых состояний H [Batch, SeqLen, HiddenSize].</param>
        /// <param name="workspaceBuf">Низкоуровневый буфер OpenCL с сохраненными гейтами из Forward.</param>
        /// <param name="nnClGlob">Диспетчер тяжелых глобальных ядер OpenCL.</param>
        public LstmOp(
            Tensor x,
            Tensor weightW,
            Tensor weightU,
            Tensor bias,
            Tensor output,
            NOpenCL.Buffer workspaceBuf,
            NnClGlob nnClGlob)
        {
            // Регистрируем входы операции для корректного обхода графа методом BuildTopo
            // Порядок в массиве важен для однозначного сопоставления градиентов в Backward
            Inputs = new Tensor[] { x, weightW, weightU, bias };
            Output = output;

            _workspaceBuf = workspaceBuf ?? throw new ArgumentNullException(nameof(workspaceBuf));
            _nnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Вычисляет градиент функции ошибки по входам операции (BPTT).
        /// Накапливает градиенты по параметрам весов и смещений атомарно на GPU.
        /// </summary>
        /// <param name="gradOutput">Градиент функции потерь по выходу этой операции dL/dH.</param>
        /// <param name="context">Текущий рабочий контекст OpenCL.</param>
        /// <returns>Массив градиентов по входам: [dL/dX, null, null, null].</returns>
        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput), "Пришедший сверху градиент dL/dH не может быть null.");

            // Извлекаем исходные тензоры из массива входов
            Tensor x = Inputs[0];
            Tensor weightW = Inputs[1];
            Tensor weightU = Inputs[2];
            Tensor bias = Inputs[3];

            // Запускаем монолитное Си-ядро обратного прохода LSTM (BPTT)
            // Оно вернет gradX и атомарно внутри VRAM обновит буферы weightW.Grad и weightU.Grad
            Tensor gradX = _nnClGlob.LstmBackward(gradOutput, x, weightW, weightU,bias, Output, _workspaceBuf);

            // Освобождаем промежуточный буфер гейтов сразу после вычисления градиентов.
            // Он больше не понадобится в текущей эпохе обучения.
            DisposeWorkspace();

            // ВАЖНО: Движок автодифференцирования AutoGrad.Backward ожидает массив 
            // градиентов по количеству элементов в массиве Inputs. 
            // Поскольку градиенты для обучаемых весов (W, U, Bias) наше монолитное ядро УЖЕ 
            // записало и просуммировало атомарно прямо во VRAM (в соответствующие .Grad поля), 
            // мы возвращаем для них null, чтобы избежать повторного логарифмического суммирования 
            // методом AccumulateGradient на хосте.

            Tensor[] inputGrads = new Tensor[Inputs.Length];
            inputGrads[0] = gradX;  // Передаем градиент для входа X, который потечет дальше по графу
            inputGrads[1] = null;   // Веса W (уже саккумулированы атомарно)
            inputGrads[2] = null;   // Веса U (уже саккумулированы атомарно)
            inputGrads[3] = null;   // Смещение Bias (уже саккумулировано атомарно в ядре)

            return inputGrads;
            //return new[]{ gradX};
        }

        /// <summary>
        /// Принудительная очистка буфера промежуточных данных.
        /// </summary>
        private void DisposeWorkspace()
        {
            if (_workspaceBuf != null)
            {
                try
                {
                    _workspaceBuf.Dispose();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Ошибка при освобождении LSTM Workspace буфера: {ex.Message}");
                }
                finally
                {
                    _workspaceBuf = null;
                }
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            DisposeWorkspace();
            _isDisposed = true;
        }
    }
    public class LstmHidOp : Op, IDisposable
    {
        private readonly NnClGlob _nnClGlob;
        private NOpenCL.Buffer _workspaceBuf;
        private bool _isDisposed = false;

        public LstmHidOp(Tensor x, Tensor weightW, Tensor weightU, Tensor bias, Tensor hPrev, Tensor cPrev,
                         Tensor output, NOpenCL.Buffer workspaceBuf, NnClGlob nnClGlob)
        {
            // Inputs теперь включает в себя старые состояния, чтобы AutoGrad прокинул ошибку назад по эпохам
            Inputs = new Tensor[] { x, weightW, weightU, bias, hPrev, cPrev };
            Output = output;
            _workspaceBuf = workspaceBuf;
            _nnClGlob = nnClGlob;
        }

        public override Tensor[] Backward(Tensor gradOutput, Context context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput), "Пришедший сверху градиент dL/dH не может быть null.");

            // Извлекаем исходные тензоры строго по индексам из массива Inputs
            Tensor x = Inputs[0];
            Tensor weightW = Inputs[1];
            Tensor weightU = Inputs[2];
            Tensor bias = Inputs[3]; // <-- Достаем наш сохраненный в графе bias
            Tensor hPrev = Inputs[4];
            Tensor cPrev = Inputs[5];

            // Вызываем исправленный метод, передавая туда bias
            var coreGrads = _nnClGlob.LstmHidBackward(gradOutput, x, weightW, weightU, bias, hPrev, cPrev, Output, _workspaceBuf);

            // Уничтожаем временный буфер гейтов во VRAM для предотвращения утечек
            if (_workspaceBuf != null)
            {
                _workspaceBuf.Dispose();
                _workspaceBuf = null;
            }

            // Собираем массив градиентов для движка автодифференцирования
            Tensor[] inputGrads = new Tensor[Inputs.Length];
            inputGrads[0] = coreGrads.gradX;     // dL/dX -> течет дальше назад по графу к предыдущим слоям
            inputGrads[1] = null;                // dL/dW (Уже внутри VRAM атомарно саккумулирован в ядре)
            inputGrads[2] = null;                // dL/dU (Уже внутри VRAM атомарно саккумулирован в ядре)
            inputGrads[3] = null;                // dL/dBias (Уже внутри VRAM атомарно саккумулирован в ядре)
            inputGrads[4] = coreGrads.gradHPrev; // dL/dH_prev -> перетекает в предыдущую временную эпоху (прошлый чанк)
            inputGrads[5] = coreGrads.gradCPrev; // dL/dC_prev -> перетекает в предыдущую временную эпоху (прошлый чанк)

            return inputGrads;
        }


        public void Dispose() { if (!_isDisposed) { _workspaceBuf?.Dispose(); _isDisposed = true; } }
    }

}
