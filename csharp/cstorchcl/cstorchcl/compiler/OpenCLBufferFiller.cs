using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.compiler
{
    /// <summary>
    /// Статический утилитный класс для быстрого обнуления OpenCL-буферов силами GPU.
    /// </summary>
    public static class OpenCLBufferFiller
    {
        private static readonly ConcurrentDictionary<Context, Kernel> _zeroKernels = new();
        private static readonly object _execLock = new();

        private const string ZeroKernelSource = @"
            kernel void fill_zero_vec16(
                global uint4* buf_vec,
                global uchar* buf_bytes,
                const long num_vec16,
                const long total_bytes
            ) {
                long gid = get_global_id(0);

                // Основной проход: пишем по 16 байт (4 float / 4 int) за раз
                if (gid < num_vec16) {
                    buf_vec[gid] = (uint4)(0u, 0u, 0u, 0u);
                }

                // Хвост: если размер буфера в байтах не кратен 16
                long tail_offset = num_vec16 * 16L + gid;
                if (tail_offset < total_bytes) {
                    buf_bytes[tail_offset] = (uchar)0;
                }
            }
        ";

        /// <summary>
        /// Заполняет OpenCL-буфер нулями на GPU.
        /// </summary>
        /// <param name="buffer">Целевой буфер в памяти GPU.</param>
        /// <param name="sizeInBytes">Размер буфера в байтах (например, tensor.Size * sizeof(float)).</param>
        public static unsafe void FillZero(NOpenCL.Buffer buffer, long sizeInBytes)
        {
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if (sizeInBytes <= 0) return;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            Kernel kernel = _zeroKernels.GetOrAdd(queue.Context, ctx =>
            {
                var program = compiler.JITProgramCache.GetOrCompile(ctx, queue.Device, ZeroKernelSource);
                return program.CreateKernel("fill_zero_vec16");
            });

            long numVec16 = sizeInBytes / 16L;
            long remainder = sizeInBytes % 16L;

            // Общее число логических потоков равно максимуму между числом vec16-блоков и остатком байтов
            long workItems = Math.Max(numVec16, remainder);

            const long localSize = 256;
            long globalSize = ((workItems + localSize - 1) / localSize) * localSize;

            lock (_execLock)
            {
                kernel.Arguments[0].SetValue(buffer);
                kernel.Arguments[1].SetValue(buffer);
                kernel.Arguments[2].SetValue((UIntPtr)sizeof(long), (IntPtr)(&numVec16));
                kernel.Arguments[3].SetValue((UIntPtr)sizeof(long),(IntPtr)(&sizeInBytes));

                queue.EnqueueNDRangeKernel(
                    kernel,
                    new[] { (IntPtr)globalSize },
                    new[] { (IntPtr)localSize }
                );
            }
        }
    }
}
