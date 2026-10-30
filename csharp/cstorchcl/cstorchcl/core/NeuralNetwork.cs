using System;
using System.Collections.Generic;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Базовый класс для создания нейросетевых моделей (Аналог torch.nn.Module в PyTorch).
    /// Объединяет глобальные операции, управляет параметрами и контекстом вычислений.
    /// </summary>
    public abstract class NeuralNetwork : IDisposable
    {
        /// <summary>
        /// Диспетчер тяжелых глобальных ядер OpenCL.
        /// </summary>
        protected readonly NnClGlob NnClGlob;

        /// <summary>
        /// Контекст вычислений OpenCL.
        /// </summary>
        protected readonly ComputeContext ComputeContext;

        /// <summary>
        /// Список всех обучаемых тензоров-параметров модели (веса, смещения).
        /// </summary>
        private readonly List<Tensor> _parameters = new List<Tensor>();

        private bool _isDisposed = false;

        /// <summary>
        /// Инициализирует модель, привязывая её к контексту вычислений.
        /// </summary>
        protected NeuralNetwork(ComputeContext computeContext, NnClGlob nnClGlob)
        {
            ComputeContext = computeContext ?? throw new ArgumentNullException(nameof(computeContext));
            NnClGlob = nnClGlob ?? throw new ArgumentNullException(nameof(nnClGlob));
        }

        /// <summary>
        /// Регистрирует тензор как обучаемый параметр модели.
        /// </summary>
        protected Tensor RegisterParameter(int[] shape, bool requiresGrad = true)
        {
            // Параметры модели изначально создаются с флагом RequiresGrad = true
            var param = new Tensor(shape, ComputeContext.Context, NOpenCL.MemoryFlags.ReadWrite, requiresGrad);
            _parameters.Add(param);
            return param;
        }

        /// <summary>
        /// Возвращает список всех параметров модели для оптимизатора.
        /// </summary>
        public IEnumerable<Tensor> GetParameters() => _parameters;

        /// <summary>
        /// Обнуляет градиенты всех параметров перед новым шагом обучения.
        /// </summary>
        public void ZeroGrad()
        {
            var queue = DeviceQueueRegistry.GetCurrent();
            foreach (var param in _parameters)
            {
                if (param.Grad != null)
                {
                    // Создаем временный хост-массив нулей для очистки буфера градиента
                    float[] zeroPattern = new float[param.Grad.Size];
                    for (int i = 0; i < zeroPattern.Length; i++)
                        zeroPattern[i] = 0;
                    long sizeInBytes = (long)param.Grad.Size * sizeof(float);
                    unsafe
                    {
                        fixed(float* zp = zeroPattern)
                        {
queue.EnqueueWriteBuffer(param.Grad.DataBuffer, true, 0, sizeInBytes, (IntPtr)zp);
                        }
                    }
                    
                }
            }
            queue.Finish();

            // Также очищаем хранилище промежуточных контекстов слоев LIFO
            GradientStorage.Clear();
        }

        /// <summary>
        /// Освобождает веса модели из VRAM видеокарты.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed) return;

            foreach (var param in _parameters)
            {
                param.Dispose();
            }
            _parameters.Clear();

            _isDisposed = true;
        }
    }
}
