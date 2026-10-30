using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.core
{
    /// <summary>
    /// Основной класс тензора библиотеки cstorchcl.
    /// Объединяет в себе буфер данных на GPU (OpenCL) и контекст графа автодифференцирования.
    /// </summary>
    public class Tensor : IDisposable
    {
        // =========================================================================
        // МЕТАДАННЫЕ И МАССИВЫ
        // =========================================================================

        /// <summary>
        /// Форма тензора (размеры по осям, например: [Batch, Head, SeqLen, Dim]).
        /// </summary>
        public int[] Shape { get; init; }

        /// <summary>
        /// Общее количество элементов в тензоре (произведение всех элементов Shape).
        /// </summary>
        public int Size { get; init; }

        /// <summary>
        /// Флаг, указывающий, нужно ли вычислять градиент для этого тензора (участие в обучении).
        /// </summary>
        public bool RequiresGrad { get; set; }

        // =========================================================================
        // OPENCL РЕСУРСЫ (ДАННЫЕ НА GPU)
        // =========================================================================

        /// <summary>
        /// Низкоуровневый буфер OpenCL, хранящий значения тензора (Forward pass).
        /// </summary>
        public NOpenCL.Buffer DataBuffer { get; private set; }

        /// <summary>
        /// Буфер OpenCL, хранящий градиенты тензора (Backward pass). 
        /// Инициализируется лениво только если RequiresGrad = true.
        /// </summary>
        public Tensor? Grad { get; set; }

        // =========================================================================
        // КОНТЕКСТ АВТОДИФФЕРЕНЦИРОВАНИЯ (AUTOGRAD)
        // =========================================================================

        /// <summary>
        /// Ссылка на операцию (Op), которая создала данный тензор в результате вычислений.
        /// Если null — тензор является листовым (leaf node), например, веса модели или входные данные.
        /// </summary>
        public object? Creator { get; set; } // В будущем заменим object на интерфейс IOp или базовый класс Op

        // Внутренний трекер очистки памяти
        private bool _isDisposed = false;

        // =========================================================================
        // КОНСТРУКТОРЫ
        // =========================================================================

        /// <summary>
        /// Создает экземпляр тензора с выделением памяти на GPU.
        /// </summary>
        /// <param name="shape">Форма тензора.</param>
        /// <param name="context">Текущий контекст OpenCL, в котором выделяется память.</param>
        /// <param name="flags">Флаги аллокации памяти OpenCL (по умолчанию ReadWrite).</param>
        /// <param name="requiresGrad">Флаг необходимости расчета градиентов.</param>
        public Tensor(int[] shape, Context context, MemoryFlags flags = MemoryFlags.ReadWrite, bool requiresGrad = false)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Форма тензора должна содержать хотя бы одно измерение.", nameof(shape));

            if (context == null)
                throw new ArgumentNullException(nameof(context), "Контекст OpenCL не может быть null.");

            Shape = (int[])shape.Clone();
            Size = CalculateSize(shape);
            RequiresGrad = requiresGrad;

            // Вычисляем размер буфера в байтах. Сейчас жестко завязано на float (4 байта)
            long sizeInBytes = (long)Size * sizeof(float);

            try
            {
                // Аллокация памяти на девайсе через NOpenCL
                DataBuffer = context.CreateBuffer(flags, sizeInBytes);
            }
            catch (Exception ex)
            {
                // Логируем контекст ошибки, упрощая отладку OOM (Out Of Memory) на GPU
                throw new InvalidOperationException(
                    $"Ошибка выделения GPU буфера размером {sizeInBytes} байт для формы [{string.Join(", ", Shape)}].", ex);
            }
        }
        /// <summary>
        /// Создает экземпляр тензора и сразу загружает в него данные из C# массива float[] во VRAM.
        /// </summary>
        /// <param name="data">Исходный массив данных на CPU (хосте).</param>
        /// <param name="shape">Форма тензора.</param>
        /// <param name="context">Текущий контекст OpenCL.</param>
        /// <param name="flags">Флаги памяти OpenCL (по умолчанию ReadWrite).</param>
        /// <param name="requiresGrad">Флаг необходимости расчета градиентов.</param>
        public unsafe Tensor(float[] data, int[] shape, Context context, MemoryFlags flags = MemoryFlags.ReadWrite, bool requiresGrad = false)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data), "Массив данных не может быть null.");

            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Форма тензора должна содержать хотя бы одно измерение.", nameof(shape));

            if (context == null)
                throw new ArgumentNullException(nameof(context), "Контекст OpenCL не может быть null.");

            Shape = (int[])shape.Clone();
            Size = CalculateSize(shape);
            RequiresGrad = requiresGrad;

            // Важнейшая валидация: проверяем, совпадает ли физический размер массива с логической формой
            if (data.Length != Size)
            {
                throw new ArgumentException(
                    $"Несоответствие размера данных форме тензора. Ожидалось элементов: {Size}, получено: {data.Length}.", nameof(data));
            }

            long sizeInBytes = (long)Size * sizeof(float);

            try
            {
                // Для оптимизации пересылки мы можем задействовать флаг CopyHostPtr, 
                // если API OpenCL поддерживает прямую аллокацию с копированием.
                // Но для гарантированной стабильности и гибкости флагов (например, если передали ReadOnly),
                // мы выделяем чистый буфер и сразу пишем в него через текущую глобальную очередь.
                DataBuffer = context.CreateBuffer(flags, sizeInBytes);

                // Получаем текущую очередь команд из регистра для немедленной записи данных
                CommandQueue queue = DeviceQueueRegistry.GetCurrent();

                // Используем fixed-указатель для безопасного получения нативного адреса управляемого массива
                fixed (float* pData = data)
                {
                    // Синхронно пишем данные во VRAM (blocking_write = true)
                    queue.EnqueueWriteBuffer(DataBuffer, true, 0, sizeInBytes, (IntPtr)pData);
                }

                // Финализируем операцию записи на устройстве
                queue.Finish();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Ошибка инициализации GPU буфера данными для формы [{string.Join(", ", Shape)}].", ex);
            }
        }

        // =========================================================================
        // ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
        // =========================================================================

        /// <summary>
        /// Вычисляет общее количество элементов на основе формы тензора.
        /// </summary>
        private static int CalculateSize(int[] shape)
        {
            int total = 1;
            foreach (var dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException($"Размерность оси не может быть меньше или равна нулю: {dim}");
                total *= dim;
            }
            return total;
        }

        /// <summary>
        /// Инициализирует буфер градиента (если он еще не создан и RequiresGrad = true).
        /// </summary>
        /// <param name="context">Контекст OpenCL.</param>
        public void InitGradient(Context context)
        {
            // Исправление 1: Выходим, если расчет градиентов НЕ требуется
            if (!RequiresGrad) return;

            // Исправление 2: Правильная проверка nullable-объекта Tensor на null
            if (Grad == null)
            {
                // Градиент всегда имеет ту же форму, что и сам тензор
                Grad = new Tensor(Shape, context, MemoryFlags.ReadWrite, requiresGrad: false);

                // ВАЖНО: В Autograd градиенты перед началом обратного прохода обнуляются.
                // Заполнение нулями (Zero-init) должно происходить через OpenCL Queue (команда fill или запись массива).
            }
        }


        // =========================================================================
        // УПРАВЛЕНИЕ ПАМЯТЬЮ (ИСПРАВЛЕНИЕ УТЕЧЕК)
        // =========================================================================

        /// <summary>
        /// Освобождает неуправляемые ресурсы OpenCL буфера.
        /// Крайне важно для предотвращения утечек VRAM на GPU.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_isDisposed) return;

            if (disposing)
            {
                // Освобождаем управляемый объект градиента, если он есть
                if (Grad != null)
                {
                    Grad.Dispose();
                    Grad = null;
                }
            }

            // Освобождаем низкоуровневый буфер OpenCL данных
            if (DataBuffer != null)
            {
                try
                {
                    DataBuffer.Dispose();
                }
                catch (Exception ex)
                {
                    // Запись в диагностику, если буфер уже был освобожден контекстом
                    System.Diagnostics.Debug.WriteLine($"Предупреждение при Dispose DataBuffer: {ex.Message}");
                }
            }

            _isDisposed = true;
        }

        /// <summary>
        /// Деструктор на случай, если пользователь забыл вызвать Dispose вручную.
        /// </summary>
        ~Tensor()
        {
            Dispose(false);
        }
    }
}
