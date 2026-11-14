using System;
using System.Collections.Concurrent;
using NOpenCL;

namespace cstorchcl.compiler
{
    /// <summary>
    /// Менеджер кэширования и переиспользования временных GPU-буферов OpenCL (Scratchpad Pool).
    /// Устраняет накладные расходы драйвера на аллокацию/освобождение VRAM на каждом батче.
    /// </summary>
    public static class OpenCLBufferManager
    {
        private record struct BufferEntry(long Size, NOpenCL.Buffer Buffer, Context Context);

        private static readonly ConcurrentDictionary<string, BufferEntry> _cache = new();
        private static readonly object _syncLock = new();

        /// <summary>
        /// Возвращает существующий буфер из кэша, если его размер >= bufSize и совпадает контекст.
        /// В противном случае освобождает старый буфер и создает новый.
        /// </summary>
        public static NOpenCL.Buffer GetOrCreateBuffer(Context context, string bufName, long bufSize,MemoryFlags memAccess=MemoryFlags.ReadWrite)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (string.IsNullOrEmpty(bufName)) throw new ArgumentException("Имя буфера не может быть пустым.", nameof(bufName));
            if (bufSize <= 0) throw new ArgumentOutOfRangeException(nameof(bufSize), "Размер буфера должен быть больше нуля.");

            // Быстрый путь без блокировки (Fast Path)
            if (_cache.TryGetValue(bufName, out var entry))
            {
                if (ReferenceEquals(entry.Context, context) && entry.Size >= bufSize)
                {
                    /*if (entry.Buffer == null || entry.Buffer.Size == 0)
                    {
                        // _cache[bufName].Buffer = null;
                        _cache.Remove(bufName, out BufferEntry r);
                        return GetOrCreateBuffer(context, bufName, bufSize,memAccess);
                    }*/
                    return entry.Buffer;
                }
            }

            // Медленный путь с синхронизацией при создании или расширении буфера
            lock (_syncLock)
            {
                if (_cache.TryGetValue(bufName, out entry))
                {
                    if (ReferenceEquals(entry.Context, context) && entry.Size >= bufSize)
                    {
                        return entry.Buffer;
                    }

                    // Старый буфер мал или принадлежит другому контексту — освобождаем VRAM
                    entry.Buffer.Dispose();
                }

                var newBuffer = context.CreateBuffer(memAccess, bufSize);
                _cache[bufName] = new BufferEntry(bufSize, newBuffer, context);
                return newBuffer;
            }
        }

        /// <summary>
        /// Освобождает конкретный буфер по имени и удаляет его из кэша.
        /// </summary>
        public static void ReleaseBuffer(string bufName)
        {
            lock (_syncLock)
            {
                if (_cache.TryRemove(bufName, out var entry))
                {
                    entry.Buffer.Dispose();
                }
            }
        }

        /// <summary>
        /// Полностью очищает пул и освобождает всю закэшированную VRAM.
        /// Вызывать при завершении работы ComputeContext.
        /// </summary>
        public static void DisposeAll()
        {
            lock (_syncLock)
            {
                foreach (var kvp in _cache)
                {
                    kvp.Value.Buffer.Dispose();
                }
                _cache.Clear();
            }
        }
    }
}
