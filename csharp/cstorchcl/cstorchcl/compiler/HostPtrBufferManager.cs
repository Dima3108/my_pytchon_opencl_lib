using NOpenCL;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

using System;

using System.Threading.Tasks;

namespace cstorchcl.compiler
{
    public class HostPtrBufferManager
    {
        private record struct BufferEntry(long Size, IntPtr Buffer);
        private static readonly ConcurrentDictionary<string, BufferEntry> _cache = new();
        private static readonly object _syncLock = new();

        // Обязательное выравнивание по границе кэш-линии (64 байта) для Zero-Copy на Intel Xe
        private const nuint CacheLineAlignment = 64;

        public static unsafe IntPtr GetOrCreateBuffer(string bufName, int bufSize, float? initW = null)
        {
            if (string.IsNullOrEmpty(bufName))
                throw new ArgumentException("Имя буфера не может быть пустым.", nameof(bufName));
            if (bufSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(bufSize), "Размер буфера должен быть больше нуля.");

            // Округляем размер вверх до кратного 64 байтам (требование Intel OpenCL для UseHostPtr)
            nuint alignedSize = (nuint)((bufSize + 63) & ~63);

            // Быстрый путь без блокировки (Fast Path)
            if (_cache.TryGetValue(bufName, out var entry))
            {
                if (entry.Size >= (long)alignedSize)
                {
                    if (entry.Buffer == IntPtr.Zero)
                    {
                        _cache.TryRemove(bufName, out _);
                        return GetOrCreateBuffer(bufName, bufSize, initW);
                    }
                    return entry.Buffer;
                }
            }

            // Медленный путь с синхронизацией при создании или расширении буфера
            lock (_syncLock)
            {
                if (_cache.TryGetValue(bufName, out entry))
                {
                    if (entry.Size >= (long)alignedSize && entry.Buffer != IntPtr.Zero)
                    {
                        return entry.Buffer;
                    }

                    if (entry.Buffer != IntPtr.Zero)
                    {
                        NativeMemory.AlignedFree((void*)entry.Buffer);
                    }
                }

                // Кроссплатформенное выделение памяти с 64-байтным выравниванием
                void* rawPtr = NativeMemory.AlignedAlloc(alignedSize, CacheLineAlignment);
                IntPtr newBuffer = (IntPtr)rawPtr;

                if (initW != null)
                {
                    float val = initW.Value;
                    int floatCount = bufSize / sizeof(float);
                    Parallel.For(0, floatCount, i =>
                    {
                        ((float*)rawPtr)[i] = val;
                    });
                }

                _cache[bufName] = new BufferEntry((long)alignedSize, newBuffer);
                return newBuffer;
            }
        }

        public static unsafe void ReleaseBuffer(string bufName)
        {
            lock (_syncLock)
            {
                if (_cache.TryRemove(bufName, out var entry) && entry.Buffer != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree((void*)entry.Buffer);
                }
            }
        }

        public static unsafe void DisposeAll()
        {
            lock (_syncLock)
            {
                foreach (var kvp in _cache)
                {
                    if (kvp.Value.Buffer != IntPtr.Zero)
                    {
                        NativeMemory.AlignedFree((void*)kvp.Value.Buffer);
                    }
                }
                _cache.Clear();
            }
        }
    }
}
