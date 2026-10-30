using System;
using System.Threading;
using NOpenCL;

namespace cstorchcl.core
{
    /// <summary>
    /// Потокобезопасный реестр для хранения и получения текущей активной очереди команд OpenCL (CommandQueue).
    /// </summary>
    public static class DeviceQueueRegistry
    {
        // Использование ThreadLocal позволяет разным потокам CPU иметь свои изолированные очереди, 
        // если в будущем вы решите делать многопоточный инференс или обучение.
        private static readonly ThreadLocal<CommandQueue> _currentQueue = new ThreadLocal<CommandQueue>();

        /// <summary>
        /// Устанавливает глобальную активную очередь команд для текущего потока.
        /// </summary>
        public static void RegisterCurrent(CommandQueue queue)
        {
            _currentQueue.Value = queue ?? throw new ArgumentNullException(nameof(queue), "Очередь команд не может быть null.");
        }

        /// <summary>
        /// Возвращает текущую активную очередь команд. 
        /// Бросает исключение, если реестр не был инициализирован.
        /// </summary>
        public static CommandQueue GetCurrent()
        {
            var queue = _currentQueue.Value;
            if (queue == null)
            {
                throw new InvalidOperationException(
                    "Очередь команд OpenCL не зарегистрирована в текущем потоке. " +
                    "Вызовите DeviceQueueRegistry.RegisterCurrent(queue) перед началом вычислений.");
            }
            return queue;
        }

        /// <summary>
        /// Очищает зарегистрированную очередь.
        /// </summary>
        public static void Clear()
        {
            _currentQueue.Value = null;
        }
    }
}
