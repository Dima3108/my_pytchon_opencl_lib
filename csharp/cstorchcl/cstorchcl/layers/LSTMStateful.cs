using cstorchcl.compiler;
using NOpenCL;
using System;

namespace cstorchcl.core
{
    /// <summary>
    /// Самостоятельный рекуррентный слой LSTM с персистентным (сохраняемым) состоянием.
    /// Автоматически удерживает и обновляет контекст (H и C) во VRAM между вызовами Forward.
    /// </summary>
    public class LSTMStateful : NeuralNetwork
    {
        public Tensor WeightW { get; private set; } // [1, InFeatures, 4 * HiddenSize]
        public Tensor WeightU { get; private set; } // [1, HiddenSize, 4 * HiddenSize]
        public Tensor Bias { get; private set; }    // [1, 4 * HiddenSize]

        // Внутренние скрытые состояния, удерживающие контекст последовательности
        private Tensor _hState; // [BatchSize, HiddenSize]
        private Tensor _cState; // [BatchSize, HiddenSize]

        private readonly int _inFeatures;
        private readonly int _hiddenSize;
        private readonly int _batchSize;
        private bool _isStateInitialized = false;

        public LSTMStateful(ComputeContext computeContext, NnClGlob nnClGlob, int batchSize, int inFeatures, int hiddenSize)
            : base(computeContext, nnClGlob)
        {
            _batchSize = batchSize;
            _inFeatures = inFeatures;
            _hiddenSize = hiddenSize;

            int totalGateFeatures = 4 * _hiddenSize;

            // 1. Выделение памяти под обучаемые параметры (веса гейтов)
            WeightW = RegisterParameter(new[] { 1, _inFeatures, totalGateFeatures }, requiresGrad: true);
            WeightU = RegisterParameter(new[] { 1, _hiddenSize, totalGateFeatures }, requiresGrad: true);
            Bias = RegisterParameter(new[] { 1, totalGateFeatures }, requiresGrad: true);

            // 2. Выделение памяти под скрытые состояния (пока заполняем нулями)
            ResetStates();

            // 3. Инициализация весов (Xavier / Glorot)
            InitializeParameters();
        }

        /// <summary>
        /// Выполняет прямой проход по текущему чанку данных. 
        /// Состояние (H, C) обновляется внутри слоя автоматически и пробрасывается в граф Autograd.
        /// </summary>
        /// <param name="x">Текущий чанк данных формы [Batch, SeqLen, InFeatures]</param>
        /// <returns>Выходные скрытые состояния чанка H [Batch, SeqLen, HiddenSize]</returns>
        public Tensor Forward(Tensor x)
        {
            // Строгая валидация конфигурации батча и признаков
            if (x.Shape[0] != _batchSize)
            {
                throw new ArgumentException($"Критическая ошибка LSTMStateful: Ожидался батч размера {_batchSize}, получен {x.Shape[0]}. " +
                    $"Для динамического батча используйте обычный класс LSTM.");
            }
            if (x.Shape[x.Shape.Length - 1] != _inFeatures)
            {
                throw new ArgumentException($"Несоответствие размерностей входа. Ожидалось {_inFeatures} признаков.");
            }

            NOpenCL.Buffer workspaceBuf;
            Tensor nextH;
            Tensor nextC;

            // Вызываем наше монолитное Си-ядро из NnClGlob, явно передавая текущие внутренние состояния
            Tensor outputH = NnClGlob.LstmWithWithState(
                x, WeightW, WeightU, Bias, _hState, _cState, _hiddenSize,
                out nextH, out nextC, out workspaceBuf
            );

            // Регистрируем операцию в графе автодифференцирования, если идет обучение
            if (outputH.RequiresGrad)
            {
                // Передаем текущие _hState и _cState в узел графа
                outputH.Creator = new LstmHidOp(x, WeightW, WeightU, Bias, _hState, _cState, outputH, workspaceBuf, NnClGlob);
            }
            else
            {
                workspaceBuf.Dispose();
            }

            // ВАЖНО: Обновляем внутренние скрытые состояния слоя для следующего чанка данных (Forward вызова)
            // Старые тензоры состояний не уничтожаем вручную, если они привязаны к графу, Autograd очистит их сам.
            _hState = nextH;
            _cState = nextC;

            return outputH;
        }

        /// <summary>
        /// Полностью сбрасывает долговременную и кратковременную память слоя (зануляет скрытые состояния VRAM).
        /// Обязательно вызывать перед подачей в сеть новой независимой текстовой или числовой последовательности!
        /// </summary>
        public void ResetStates()
        {
            // Если состояния уже были созданы, освобождаем старые буферы во VRAM
            if (_isStateInitialized)
            {
                _hState?.Dispose();
                _cState?.Dispose();
            }

            int[] stateShape = { _batchSize, _hiddenSize };
            float[] zeroPattern = new float[_batchSize * _hiddenSize]; // По умолчанию заполнен нулями C#

            // Создаем чистые тензоры состояний. Они не требуют расчета градиентов по умолчанию, 
            // так как являются начальными константными точками отсчета (листьями графа).
            _hState = new Tensor(zeroPattern, stateShape, ComputeContext.Context, "LSTMStateful_hSatete", MemoryFlags.ReadWrite, requiresGrad: false);
            _cState = new Tensor(zeroPattern, stateShape, ComputeContext.Context, "LSTMStateful_cSatete", MemoryFlags.ReadWrite, requiresGrad: false);

            _isStateInitialized = true;
        }

        /// <summary>
        /// Стандартная инициализация весов (Xavier) и Bias (Forget Bias = 1.0f).
        /// </summary>
        private void InitializeParameters()
        {
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            var rand = new Random();
            int totalGateFeatures = 4 * _hiddenSize;

            float boundW = (float)(Math.Sqrt(6.0) / Math.Sqrt(_inFeatures + _hiddenSize));
            float[] hostWeightsW = new float[WeightW.Size];
            for (int i = 0; i < hostWeightsW.Length; i++)
                hostWeightsW[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundW;

            float boundU = (float)(Math.Sqrt(6.0) / Math.Sqrt(_hiddenSize + _hiddenSize));
            float[] hostWeightsU = new float[WeightU.Size];
            for (int i = 0; i < hostWeightsU.Length; i++)
                hostWeightsU[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundU;

            float[] hostBias = new float[Bias.Size];
            int forgetGateStart = _hiddenSize;
            int forgetGateEnd = 2 * _hiddenSize;
            for (int i = 0; i < hostBias.Length; i++)
                hostBias[i] = (i >= forgetGateStart && i < forgetGateEnd) ? 1.0f : 0.0f;

            unsafe
            {
                fixed (float* hw = hostWeightsW, hu = hostWeightsU, hb = hostBias)
                {
                    queue.EnqueueWriteBuffer(WeightW.DataBuffer, true, 0, (long)WeightW.Size * sizeof(float), (IntPtr)hw);
                    queue.EnqueueWriteBuffer(WeightU.DataBuffer, true, 0, (long)WeightU.Size * sizeof(float), (IntPtr)hu);
                    queue.EnqueueWriteBuffer(Bias.DataBuffer, true, 0, (long)Bias.Size * sizeof(float), (IntPtr)hb);
                }
            }
            queue.Finish();
        }

        public  void Dispose()
        {
            base.Dispose();
            if (_isStateInitialized)
            {
                _hState?.Dispose();
                _cState?.Dispose();
            }
        }
    }
}
