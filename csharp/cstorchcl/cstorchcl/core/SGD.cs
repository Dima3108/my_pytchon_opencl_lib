using System;
using System.Collections.Generic;
using NOpenCL;
using cstorchcl.compiler;

namespace cstorchcl.core
{
    /// <summary>
    /// Оптимизатор стохастического градиентного спуска (SGD) на OpenCL.
    /// </summary>
    public class SGD
    {
        private readonly IEnumerable<Tensor> _parameters;
        private readonly float _learningRate;
        private readonly ComputeContext _computeCtx;

        public SGD(IEnumerable<Tensor> parameters, float learningRate, ComputeContext computeCtx)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            _learningRate = learningRate;
            _computeCtx = computeCtx ?? throw new ArgumentNullException(nameof(computeCtx));
        }

        /// <summary>
        /// Шаг оптимизации: обновляет веса всех параметров модели на основе рассчитанных градиентов.
        /// </summary>
        public void Step()
        {
            string kernelSource = @"
                kernel void sgd_step(global float* W, global const float* G) {
                    int id = get_global_id(0);
                    W[id] -= "+_learningRate.ToString().Replace(",",".")+@" * G[id];
                }";

            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);
            using var kernel = prg.CreateKernel("sgd_step");
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            foreach (var param in _parameters)
            {
                // Обновляем только те параметры, для которых был рассчитан градиент
                if (param.RequiresGrad && param.Grad != null)
                {
                    kernel.Arguments[0].SetValue(param.DataBuffer);
                    kernel.Arguments[1].SetValue(param.Grad.DataBuffer);
                    //kernel.Arguments.SetValue(_learningRate);

                    queue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(param.Size) }, null);
                }
            }
            queue.Finish();
        }
    }
}
