using System;
using System.Collections.Generic;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;
using Buffer = NOpenCL.Buffer;
using cstorchcl.extensions;
using System.Collections.Immutable;
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
        private readonly ComputeContext _ComputeContext;
        public Adam(ComputeContext context,List<Tensor> parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            _lr = lr;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _ComputeContext = context;
            // Берем контекст из первого доступного тензора
            if (_parameters.Count == 0)
                throw new ArgumentException("Список параметров для оптимизации не должен быть пустым.");

            // Здесь мы задействуем твою систему очередей и JIT-кэш
            var queue = DeviceQueueRegistry.GetCurrent();
            _context = queue.Context;

            // Загружаем или компилируем ядро для Adam
            // (В реальном коде можно читать из файла cl_kernels/adam_step.cl, как у тебя в NnClGlob)
            string kernelSource = // В конструкторе Adam компилируем два ядра: adam_step_1 и adam_step_8
 @"
kernel void adam_step_1(
    global float* w0, global const float* g0, global float* m0, global float* v0, const int s0,
    const float lr, const float b1, const float b2, const float eps, const float bc1, const float bc2)
{
    int id = get_global_id(0);
    if (id < s0) {
        float g = g0[id];
        float m = b1 * m0[id] + (1.0f - b1) * g;
        float v = b2 * v0[id] + (1.0f - b2) * (g * g);
        m0[id] = m; v0[id] = v;
        w0[id] -= (lr / (sqrt(v / bc2) + eps)) * (m / bc1);
    }
}

#define ADAM_UPDATE(w, g, m, v, sz) \
    if (id < sz) { \
        float gv = g[id]; \
        float mv = b1 * m[id] + (1.0f - b1) * gv; \
        float vv = b2 * v[id] + (1.0f - b2) * (gv * gv); \
        m[id] = mv; v[id] = vv; \
        w[id] -= (lr / (sqrt(vv / bc2) + eps)) * (mv / bc1); \
    }

kernel void adam_step_8(
    global float* w0, global const float* g0, global float* m0, global float* v0, const int s0,
    global float* w1, global const float* g1, global float* m1, global float* v1, const int s1,
    global float* w2, global const float* g2, global float* m2, global float* v2, const int s2,
    global float* w3, global const float* g3, global float* m3, global float* v3, const int s3,
    global float* w4, global const float* g4, global float* m4, global float* v4, const int s4,
    global float* w5, global const float* g5, global float* m5, global float* v5, const int s5,
    global float* w6, global const float* g6, global float* m6, global float* v6, const int s6,
    global float* w7, global const float* g7, global float* m7, global float* v7, const int s7,
    const float lr, const float b1, const float b2, const float eps, const float bc1, const float bc2)
{
    int id = get_global_id(0);
    ADAM_UPDATE(w0, g0, m0, v0, s0);
    ADAM_UPDATE(w1, g1, m1, v1, s1);
    ADAM_UPDATE(w2, g2, m2, v2, s2);
    ADAM_UPDATE(w3, g3, m3, v3, s3);
    ADAM_UPDATE(w4, g4, m4, v4, s4);
    ADAM_UPDATE(w5, g5, m5, v5, s5);
    ADAM_UPDATE(w6, g6, m6, v6, s6);
    ADAM_UPDATE(w7, g7, m7, v7, s7);
}";
            ;

            _adamProgram = JITProgramCache.GetOrCompile(_context, queue.Device, kernelSource);
            //_adamKernel = _adamProgram.CreateKernel("adam_step");
            _adamKernel1 = _adamProgram.CreateKernel("adam_step_1");
            _adamKernel8 = _adamProgram.CreateKernel("adam_step_8");
        }
        private Kernel _adamKernel1, _adamKernel8;
        /// <summary>
        /// Выполняет один шаг обновления весов для всех зарегистрированных параметров.
        /// </summary>
        public void Step()
        {
            _t++;
            float bc1 = (float)(1.0 - Math.Pow(_beta1, _t));
            float bc2 = (float)(1.0 - Math.Pow(_beta2, _t));
            var queue = DeviceQueueRegistry.GetCurrent();

            // 1. Собираем активные параметры и лениво инициализируем m/v (только на 1-м шаге)
            var active = new List<(Tensor Param, Buffer M, Buffer V)>(_parameters.Count);
            for (int i = 0; i < _parameters.Count; i++)
            {
                var p = _parameters[i];
                if (!p.RequiresGrad || p.Grad == null) continue;

                if (!_mStates.TryGetValue(p, out var mBuf))
                {
                    long bytes = (long)p.Size * sizeof(float);
                    mBuf = OpenCLBufferManager.GetOrCreateBuffer(queue.Context, $"adam_m_{i}_{p.TensorID}", bytes);
                    var vBuf = OpenCLBufferManager.GetOrCreateBuffer(queue.Context, $"adam_v_{i}_{p.TensorID}", bytes);
                    OpenCLBufferFiller.FillZero(mBuf, bytes);
                    OpenCLBufferFiller.FillZero(vBuf, bytes);
                    _mStates[p] = mBuf;
                    _vStates[p] = vBuf;
                }
                active.Add((p, _mStates[p], _vStates[p]));
            }

            int idx = 0;
            const int localSize = 256;

            // 2. Обновляем пачками по 8 тензоров за 1 запуск ядра!
            while (idx + 8 <= active.Count)
            {
                int arg = 0;
                long maxLen = 0;
                for (int k = 0; k < 8; k++)
                {
                    var item = active[idx + k];
                    _adamKernel8.Arguments[arg++].SetValue(item.Param.DataBuffer);
                    _adamKernel8.Arguments[arg++].SetValue(item.Param.Grad.DataBuffer);
                    _adamKernel8.Arguments[arg++].SetValue(item.M);
                    _adamKernel8.Arguments[arg++].SetValue(item.V);
                    _adamKernel8.Arguments[arg++].SetValue(item.Param.Size);
                    if (item.Param.Size > maxLen) maxLen = item.Param.Size;
                }
                _adamKernel8.Arguments[arg++].SetValue(_lr);
                _adamKernel8.Arguments[arg++].SetValue(_beta1);
                _adamKernel8.Arguments[arg++].SetValue(_beta2);
                _adamKernel8.Arguments[arg++].SetValue(_epsilon);
                _adamKernel8.Arguments[arg++].SetValue(bc1);
                _adamKernel8.Arguments[arg++].SetValue(bc2);

                long globalSize = ((maxLen + localSize - 1) / localSize) * localSize;
                queue.EnqueueNDRangeKernel(_adamKernel8, null, new[] { (IntPtr)globalSize }, new[] { (IntPtr)localSize });
                idx += 8;
            }

            // 3. Хвост (оставшиеся < 8 тензоров) обновляем по одному
            while (idx < active.Count)
            {
                var item = active[idx++];
                int arg = 0;
                _adamKernel1.Arguments[arg++].SetValue(item.Param.DataBuffer);
                _adamKernel1.Arguments[arg++].SetValue(item.Param.Grad.DataBuffer);
                _adamKernel1.Arguments[arg++].SetValue(item.M);
                _adamKernel1.Arguments[arg++].SetValue(item.V);
                _adamKernel1.Arguments[arg++].SetValue(item.Param.Size);
                _adamKernel1.Arguments[arg++].SetValue(_lr);
                _adamKernel1.Arguments[arg++].SetValue(_beta1);
                _adamKernel1.Arguments[arg++].SetValue(_beta2);
                _adamKernel1.Arguments[arg++].SetValue(_epsilon);
                _adamKernel1.Arguments[arg++].SetValue(bc1);
                _adamKernel1.Arguments[arg++].SetValue(bc2);

                long globalSize = ((item.Param.Size + localSize - 1) / localSize) * localSize;
                queue.EnqueueNDRangeKernel(_adamKernel1, null, new[] { (IntPtr)globalSize }, new[] { (IntPtr)localSize });
            }
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
              //  float[] zeroPattern = new float[param.Grad.Size];
               long bytesSize = (long)param.Grad.Size * sizeof(float);
                unsafe
                {
                    //  fixed (float* zPtr = zeroPattern)
                   // float* zPtr = (float*)HostPtrBufferManager.GetOrCreateBuffer($"zeroBuffer{param.Grad.Size}",(int)bytesSize,0);
                    {
                       // queue.EnqueueWriteBuffer(param.Grad.DataBuffer, true, 0, bytesSize, (IntPtr)zPtr);
                    }
                    OpenCLBufferFiller.FillZero(param.Grad.DataBuffer, bytesSize);
                }
            }
            //queue.Finish();
        }
    }
}
