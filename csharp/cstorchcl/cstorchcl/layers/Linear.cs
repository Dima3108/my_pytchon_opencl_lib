using cstorchcl.compiler;
using NOpenCL;
using System;
using System.Collections.Generic;

namespace cstorchcl.core
{
    /// <summary>
    /// Полносвязный (линейный) слой нейросети с поддержкой JIT-сцепки и Broadcasting смещения на GPU.
    /// </summary>
    public class Linear : NeuralNetwork
    {
        /// <summary>
        /// Обучаемая матрица весов слоя. Форма: [1, InFeatures, OutFeatures]
        /// </summary>
        public Tensor Weight { get; private set; }

        /// <summary>
        /// Обучаемый вектор смещения (bias). Форма: [1, OutFeatures] — будет автоматически растягиваться по батчу.
        /// </summary>
        public Tensor Bias { get; private set; }

        private readonly int _inFeatures;
        private readonly int _outFeatures;

        /// <summary>
        /// Инициализирует линейный слой и заполняет его веса случайным шумом для разрушения симметрии.
        /// </summary>
        public Linear(ComputeContext computeContext, NnClGlob nnClGlob, int inFeatures, int outFeatures)
            : base(computeContext, nnClGlob)
        {
            _inFeatures = inFeatures;
            _outFeatures = outFeatures;

            // 1. Аллокация памяти во VRAM девайса
            // Веса делаем 3D формы [1, K, N] под требования нашего пакетного ядра BMM
            Weight = RegisterParameter(new[] { 1, inFeatures, outFeatures }, requiresGrad: true);
            // Bias делаем 2D формы [1, OutFeatures] под требования Broadcasting
            Bias = RegisterParameter(new[] { 1, outFeatures }, requiresGrad: true);

            // 2. Инициализация весов методом Ксавье (Xavier/Glorot Initialization)
            // Это стандарт глубокого обучения, предохраняющий tanh от насыщения и зануления градиентов
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            var rand = new Random();

            float bound = (float)(Math.Sqrt(6.0) / Math.Sqrt(inFeatures + outFeatures));

            // Заполнение матрицы весов W
            float[] hostWeights = new float[Weight.Size];
            for (int i = 0; i < hostWeights.Length; i++)
            {
                hostWeights[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * bound;
            }
            long weightsBytes = (long)Weight.Size * sizeof(float);
            unsafe
            {
                fixed (float* hw = hostWeights)
                {
 queue.EnqueueWriteBuffer(Weight.DataBuffer, true, 0, weightsBytes, (IntPtr)hw);
                }
            }
           

            // Заполнение вектора смещения Bias (по умолчанию заполняем нулями, как в PyTorch)
            float[] hostBias = new float[Bias.Size];
            long biasBytes = (long)Bias.Size * sizeof(float);
            unsafe
            {
                fixed (float* hb = hostBias)
                {
queue.EnqueueWriteBuffer(Bias.DataBuffer, true, 0, biasBytes, (IntPtr)hb);
                }
            }
            

            queue.Finish();
        }

        /// <summary>
        /// Выполняет высокооптимизированный прямой проход: BMM -> JIT Сложение с Bias.
        /// На вход ожидает 3D тензор данных формы [Batch, M, InFeatures].
        /// </summary>
        public Tensor Forward(Tensor x)
        {
            if (x.Shape[x.Shape.Length - 1] != _inFeatures)
            {
                throw new ArgumentException($"Несоответствие размерностей входа. Ожидалось {_inFeatures} признаков, получено {x.Shape[x.Shape.Length - 1]}.");
            }

            // Единый монолитный вызов без промежуточных аллокаций и JIT-компиляции
            return NnClGlob.LinearForward(x, Weight, Bias);
        }
    }
}
