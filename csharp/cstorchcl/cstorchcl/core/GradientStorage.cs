using System;
using System.Collections.Generic;
using NOpenCL;

namespace cstorchcl.core
{
    /// <summary>
    /// Контейнер для хранения промежуточных буферов данных конкретного слоя, 
    /// необходимых для фазы расчета градиентов (Backward pass).
    /// </summary>
    public struct LayerBuffers : IDisposable
    {
        /// <summary>
        /// Словарь исходных данных и их символических имен в ядре.
        /// </summary>
        public Dictionary<string, NOpenCL.Buffer> layerBuffers;

        public LayerBuffers()
        {
            layerBuffers = new Dictionary<string, NOpenCL.Buffer>();
        }

        /// <summary>
        /// Очищает промежуточные буферы слоя во VRAM.
        /// </summary>
        public void Dispose()
        {
            if (layerBuffers == null) return;

            // Освобождаем нативные OpenCL буферы
            foreach (var buffer in layerBuffers.Values)
            {
                buffer?.Dispose();
            }

            layerBuffers.Clear();
        }
    }

    /// <summary>
    /// Хранилище контекстов слоев для вычисления градиентов.
    /// Работает по принципу LIFO (Стек: первый пришел — последний ушел).
    /// </summary>
    public static class GradientStorage
    {
        // Стек для сохранения промежуточных данных слоев во время Forward прохода
        private static readonly Stack<LayerBuffers> _storage = new Stack<LayerBuffers>();
        private static readonly object _lock = new object();

        /// <summary>
        /// Помещает буферы слоя в стек хранения перед вызовом backward.
        /// </summary>
        public static void Push(LayerBuffers buffers)
        {
            lock (_lock)
            {
                _storage.Push(buffers);
            }
        }

        /// <summary>
        /// Извлекает последний сохраненный контекст слоя (LIFO).
        /// </summary>
        public static LayerBuffers Pop()
        {
            lock (_lock)
            {
                if (_storage.Count == 0)
                    throw new InvalidOperationException("Попытка извлечь контекст слоя из пустого GradientStorage.");
                return _storage.Pop();
            }
        }

        /// <summary>
        /// Полная очистка хранилища градиентов.
        /// </summary>
        public static void Clear()
        {
            lock (_lock)
            {
                while (_storage.Count > 0)
                {
                    var buffers = _storage.Pop();
                    buffers.Dispose();
                }
            }
        }
    }
}

