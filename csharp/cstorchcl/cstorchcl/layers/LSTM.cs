using cstorchcl.compiler;
using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.layers
{
    /// <summary>
    /// Высокоуровневый монолитный слой LSTM (Long Short-Term Memory).
    /// Автоматически управляет объединенными весами гейтов во VRAM и интегрируется в граф автодифференцирования.
    /// </summary>
    public class LSTM : NeuralNetwork
    {
        /// <summary>
        /// Объединенная матрица входных весов для всех 4 гейтов (i, f, g, o).
        /// Форма под требования BMM/ядра: [1, InFeatures, 4 * HiddenSize]
        /// </summary>
        public Tensor WeightW { get; private set; }

        /// <summary>
        /// Объединенная матрица рекуррентных весов для скрытого состояния всех 4 гейтов (i, f, g, o).
        /// Форма: [1, HiddenSize, 4 * HiddenSize]
        /// </summary>
        public Tensor WeightU { get; private set; }

        /// <summary>
        /// Объединенный вектор смещений гейтов (bias).
        /// Форма под требования Broadcasting: [1, 4 * HiddenSize]
        /// </summary>
        public Tensor Bias { get; private set; }

        private readonly int _inFeatures;
        private readonly int _hiddenSize;

        /// <summary>
        /// Инициализирует слой LSTM и заполняет его веса методами Xavier/Glorot.
        /// </summary>
        public LSTM(ComputeContext computeContext, NnClGlob nnClGlob, int inFeatures, int hiddenSize)
            : base(computeContext, nnClGlob)
        {
            _inFeatures = inFeatures;
            _hiddenSize = hiddenSize;

            int totalGateFeatures = 4 * _hiddenSize;

            // 1. Аллокация памяти во VRAM девайса с флагом RequiresGrad = true
            WeightW = RegisterParameter(new[] { 1, _inFeatures, totalGateFeatures }, requiresGrad: true);
            WeightU = RegisterParameter(new[] { 1, _hiddenSize, totalGateFeatures }, requiresGrad: true);
            Bias = RegisterParameter(new[] { 1, totalGateFeatures }, requiresGrad: true);

            // 2. Инициализация весов на CPU и пересылка во VRAM
            InitializeWeights();
        }

        /// <summary>
        /// Выполняет высокооптимизированный прямой проход LSTM через монолитное ядро.
        /// На вход ожидает 3D тензор формы [Batch, SeqLen, InFeatures].
        /// </summary>
        public Tensor Forward(Tensor x)
        {
            // Валидация входных признаков по последней оси
            if (x.Shape[x.Shape.Length - 1] != _inFeatures)
            {
                throw new ArgumentException($"Несоответствие размерностей входа LSTM. " +
                    $"Ожидалось {_inFeatures} признаков, получено {x.Shape[x.Shape.Length - 1]}.");
            }

            // Вызываем метод из NnClGlob, который аллоцирует выходной тензор H и возвращает нам GPU-буфер workspace
            NOpenCL.Buffer workspaceBuf;
            Tensor outputH = NnClGlob.Lstm(x, WeightW, WeightU, Bias, _hiddenSize, out workspaceBuf);

            // Если включен режим Autograd и требуется расчет градиента, привязываем операцию к выходу
            if (outputH.RequiresGrad)
            {
                // Передаем все входы, выход и низкоуровневый буфер в узел графа
                outputH.Creator = new LstmOp(x, WeightW, WeightU, Bias, outputH, workspaceBuf, NnClGlob);
            }
            else
            {
                // Если мы в режиме инференса (RequiresGrad = false), workspace нам больше не нужен,
                // и так как граф его не подберет, освобождаем буфер немедленно.
                workspaceBuf.Dispose();
            }

            return outputH;
        }

        /// <summary>
        /// Инициализирует объединенные матрицы весов стандартными методами глубокого обучения.
        /// </summary>
        private void InitializeWeights()
        {
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            var rand = new Random();
            int totalGateFeatures = 4 * _hiddenSize;

            // --- Инициализация WeightW (Xavier) ---
            float boundW = (float)(Math.Sqrt(6.0) / Math.Sqrt(_inFeatures + _hiddenSize));
            float[] hostWeightsW = new float[WeightW.Size];
            for (int i = 0; i < hostWeightsW.Length; i++)
            {
                hostWeightsW[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundW;
            }

            // --- Инициализация WeightU (Xavier) ---
            float boundU = (float)(Math.Sqrt(6.0) / Math.Sqrt(_hiddenSize + _hiddenSize));
            float[] hostWeightsU = new float[WeightU.Size];
            for (int i = 0; i < hostWeightsU.Length; i++)
            {
                hostWeightsU[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * boundU;
            }

            // --- Инициализация Bias ---
            // По умолчанию заполняем нулями, НО гейт забывания (Forget Gate) 
            // инициализируем единицами (1.0f), чтобы избежать затухания градиентов на старте.
            // Наша структура в памяти: [i_gate (0..H), f_gate (H..2H), g_gate (2H..3H), o_gate (3H..4H)]
            float[] hostBias = new float[Bias.Size];
            int forgetGateStart = _hiddenSize;
            int forgetGateEnd = 2 * _hiddenSize;

            for (int i = 0; i < hostBias.Length; i++)
            {
                if (i >= forgetGateStart && i < forgetGateEnd)
                    hostBias[i] = 1.0f; // Устанавливаем Forget Bias = 1.0
                else
                    hostBias[i] = 0.0f;
            }

            // --- Синхронная запись подготовленных массивов в VRAM ---
            unsafe
            {
                fixed (float* hw = hostWeightsW)
                {
                    long bytesW = (long)WeightW.Size * sizeof(float);
                    queue.EnqueueWriteBuffer(WeightW.DataBuffer, true, 0, bytesW, (IntPtr)hw);
                }

                fixed (float* hu = hostWeightsU)
                {
                    long bytesU = (long)WeightU.Size * sizeof(float);
                    queue.EnqueueWriteBuffer(WeightU.DataBuffer, true, 0, bytesU, (IntPtr)hu);
                }

                fixed (float* hb = hostBias)
                {
                    long bytesBias = (long)Bias.Size * sizeof(float);
                    queue.EnqueueWriteBuffer(Bias.DataBuffer, true, 0, bytesBias, (IntPtr)hb);
                }
            }

            queue.Finish();
        }
    }
}
