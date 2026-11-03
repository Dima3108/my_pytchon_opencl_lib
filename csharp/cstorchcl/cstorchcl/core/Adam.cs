using System;
using System.Collections.Generic;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;
using Buffer = NOpenCL.Buffer;
using cstorchcl.extensions;
namespace cstorchcl.optimizers
{
    /// <summary>
    /// Адаптивный оптимизатор Adam (Adaptive Moment Estimation) для OpenCL-тензоров.
    /// </summary>
    public class Adam
    {
        private readonly List<Tensor> _parameters;
        private readonly float _lr;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private int _t = 0; // Счетчик шагов оптимизации

        // Словари для хранения буферов первого (m) и второго (v) моментов для каждого тензора параметров
        private readonly Dictionary<Tensor, Buffer> _mStates = new Dictionary<Tensor, Buffer>();
        private readonly Dictionary<Tensor, Buffer> _vStates = new Dictionary<Tensor, Buffer>();

        private readonly Program _adamProgram;
        private readonly Kernel _adamKernel;
        private readonly Context _context;

        public Adam(List<Tensor> parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            _lr = lr;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;

            // Берем контекст из первого доступного тензора
            if (_parameters.Count == 0)
                throw new ArgumentException("Список параметров для оптимизации не должен быть пустым.");

            // Здесь мы задействуем твою систему очередей и JIT-кэш
            var queue = DeviceQueueRegistry.GetCurrent();
            _context = queue.Context;

            // Загружаем или компилируем ядро для Adam
            // (В реальном коде можно читать из файла cl_kernels/adam_step.cl, как у тебя в NnClGlob)
            string kernelSource = @"
            kernel void adam_step(global float* w, global const float* g, global float* m, global float* v,
                                  const float lr, const float b1, const float b2, const float eps,
                                  const float bc1, const float bc2, const int size) {
                int id = get_global_id(0);
                if (id >= size) return;
                float grad_val = g[id];
                m[id] = b1 * m[id] + (1.0f - b1) * grad_val;
                v[id] = b2 * v[id] + (1.0f - b2) * (grad_val * grad_val);
                float m_hat = m[id] / bc1;
                float v_hat = v[id] / bc2;
                w[id] -= (lr / (sqrt(v_hat) + eps)) * m_hat;
            }";

            _adamProgram = JITProgramCache.GetOrCompile(_context, queue.Device, kernelSource);
            _adamKernel = _adamProgram.CreateKernel("adam_step");
        }

        /// <summary>
        /// Выполняет один шаг обновления весов для всех зарегистрированных параметров.
        /// </summary>
        public void Step()
        {
            _t++; // Инкрементируем шаг для корректировки смещения

            // Рассчитываем коэффициенты коррекции смещения (bias correction) на текущем шаге
            float biasCorrection1 = (float)(1.0 - Math.Pow(_beta1, _t));
            float biasCorrection2 = (float)(1.0 - Math.Pow(_beta2, _t));

            var queue = DeviceQueueRegistry.GetCurrent();

            foreach (var param in _parameters)
            {
                // Оптимизируем только те тензоры, для которых считаются градиенты и они были накоплены
                if (!param.RequiresGrad || param.Grad == null) continue;

                int size = param.Size;

                // Если состояний моментов для этого тензора еще нет во VRAM — аллоцируем и обнуляем их
                if (!_mStates.ContainsKey(param))
                {
                    // Создаем массивы нулей для инициализации (как в твоем фиксе для EmbeddingBackward)
                    float[] zeroPattern = new float[size];
                    long bytesSize = (long)size * sizeof(float);

                    Buffer mBuf = _context.CreateBuffer(MemoryFlags.ReadWrite, bytesSize);
                    Buffer vBuf = _context.CreateBuffer(MemoryFlags.ReadWrite, bytesSize);

                    // Синхронно обнуляем буферы на устройстве
                    unsafe
                    {
                        fixed (float* zPtr = zeroPattern)
                        {
                            queue.EnqueueWriteBuffer(mBuf, true, 0, bytesSize, (IntPtr)zPtr);
                            queue.EnqueueWriteBuffer(vBuf, true, 0, bytesSize, (IntPtr)zPtr);
                        }
                    }

                    _mStates[param] = mBuf;
                    _vStates[param] = vBuf;
                }

                Buffer mBuffer = _mStates[param];
                Buffer vBuffer = _vStates[param];

                // Передаем аргументы в OpenCL ядро по индексам
                int argIdx = 0;
                _adamKernel.Arguments[argIdx++].SetValue(param.DataBuffer);      // weight
                _adamKernel.Arguments[argIdx++].SetValue(param.Grad.DataBuffer); // grad
                _adamKernel.Arguments[argIdx++].SetValue(mBuffer);               // m
                _adamKernel.Arguments[argIdx++].SetValue(vBuffer);               // v
                _adamKernel.Arguments[argIdx++].SetValue(_lr);
                _adamKernel.Arguments[argIdx++].SetValue(_beta1);
                _adamKernel.Arguments[argIdx++].SetValue(_beta2);
                _adamKernel.Arguments[argIdx++].SetValue(_epsilon);
                
                _adamKernel.Arguments[argIdx++].SetValue(biasCorrection1);
                _adamKernel.Arguments[argIdx++].SetValue(biasCorrection2);
                _adamKernel.Arguments[argIdx++].SetValue(size);

                // Запуск одномерной сетки потоков по размеру тензора
                IntPtr globalSize = new IntPtr(size);
                queue.EnqueueNDRangeKernel(_adamKernel, new[] { globalSize }, null);
            }

            // Дожидаемся завершения применения шага оптимизатора ко всем параметрам на GPU
            queue.Finish();
        }

        /// <summary>
        /// Очищает накопленные градиенты у параметров (аналог zero_grad в PyTorch).
        /// </summary>
        public void ZeroGrad()
        {
            var queue = DeviceQueueRegistry.GetCurrent();
            foreach (var param in _parameters)
            {
                if (param.Grad == null) continue;

                // Обнуляем буфер градиента через хост-массив сил .NET
                float[] zeroPattern = new float[param.Grad.Size];
                long bytesSize = (long)param.Grad.Size * sizeof(float);
                unsafe
                {
                    fixed (float* zPtr = zeroPattern)
                    {
                        queue.EnqueueWriteBuffer(param.Grad.DataBuffer, true, 0, bytesSize, (IntPtr)zPtr);
                    }
                }
            }
            queue.Finish();
        }
    }
}
