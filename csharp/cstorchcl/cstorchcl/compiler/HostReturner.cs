using System;
using System.Text;
using NOpenCL;
using cstorchcl.core;

namespace cstorchcl.compiler
{
    /// <summary>
    /// Компонент-фасад для безопасного скачивания тензорных данных и градиентов 
    /// из VRAM ускорителя в RAM процессора (CPU), а также для их отладочного вывода.
    /// </summary>
    public static class HostReturner
    {
        /// <summary>
        /// Скачивает данные прямого прохода (Forward) тензора из OpenCL VRAM в обычный C# массив float[].
        /// </summary>
        public static float[] ToCpu(Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            // Выделяем плоский массив нужного размера в оперативной памяти CPU
            float[] hostArray = new float[tensor.Size];
            long sizeInBytes = (long)tensor.Size * sizeof(float);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Читаем буфер из GPU во VRAM
            // blocking_read = true гарантирует, что поток CPU дождется окончания копирования
            unsafe
            {
                fixed(float* ha = hostArray)
                {
 queue.EnqueueReadBuffer(tensor.DataBuffer, true, 0, sizeInBytes, (IntPtr)ha);
                }
            }
           

            // Встроенная жесткая синхронизация для безопасности рантайма
            queue.Finish();

            return hostArray;
        }

        /// <summary>
        /// Скачивает данные обратного прохода (Backward) тензора из OpenCL VRAM в массив float[].
        /// </summary>
        public static float[] ToCpuGrad(Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            if (tensor.Grad == null)
            {
                throw new InvalidOperationException(
                    "У данного тензора отсутствует буфер градиентов (возможно, RequiresGrad = false " +
                    "или операция Backward еще не вызывалась).");
            }

            float[] hostGrad = new float[tensor.Grad.Size];
            long sizeInBytes = (long)tensor.Grad.Size * sizeof(float);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            unsafe
            {
                fixed(float* hg = hostGrad)
                {
queue.EnqueueReadBuffer(tensor.Grad.DataBuffer, true, 0, sizeInBytes, (IntPtr)hg);
                }
            }
            // Скачиваем данные из буфера градиента
            
            queue.Finish();

            return hostGrad;
        }

        /// <summary>
        /// Красиво выводит тензор и его градиенты в консоль для отладки, имитируя форматирование PyTorch.
        /// </summary>
        public static void PrintTensor(Tensor tensor, string name = "Tensor", bool printGrad = false)
        {
            if (tensor == null) return;

            string separator = new string('=', 60);
            Console.WriteLine(separator);
            Console.WriteLine($"ОТЛАДКА ТЕНЗОРА: {name}");
            Console.WriteLine($". Форма (Shape): [{string.Join(", ", tensor.Shape)}]");
            Console.WriteLine($". Всего элементов: {tensor.Size}");
            Console.WriteLine($". Тип данных: System.Single (float32)");
            Console.WriteLine(separator);

            // Скачиваем и выводим Forward данные
            float[] data = ToCpu(tensor);
            Console.WriteLine("Данные (Forward):");
            PrintFormattedArray(data, tensor.Shape);

            if (printGrad)
            {
                Console.WriteLine("\nГрадиенты (Backward):");
                if (tensor.Grad != null)
                {
                    float[] gradData = ToCpuGrad(tensor);
                    PrintFormattedArray(gradData, tensor.Shape);
                }
                else
                {
                    Console.WriteLine("[Градиенты отсутствуют / RequiresGrad = false]");
                }
            }
            Console.WriteLine(separator + "\n");
        }

        /// <summary>
        /// Вспомогательный метод для компактного вывода многомерных массивов (имитация edgeitems из NumPy).
        /// Выводит первые и последние элементы, скрывая середину, если массив слишком большой.
        /// </summary>
        private static void PrintFormattedArray(float[] array, int[] shape)
        {
            // Для простоты прототипа реализуем компактный вывод плоского представления с ограничением
            int maxElementsToShow = 12;

            if (array.Length <= maxElementsToShow)
            {
                Console.WriteLine($"  [{string.Join(", ", array.Select(x => x.ToString("F4")))}]");
            }
            else
            {
                var sb = new StringBuilder();
                sb.Append("  [");
                // Первые 3 элемента
                for (int i = 0; i < 3; i++) sb.Append(array[i].ToString("F4") + ", ");

                sb.Append("... , ");

                // Последние 3 элемента
                for (int i = array.Length - 3; i < array.Length; i++)
                {
                    sb.Append(array[i].ToString("F4"));
                    if (i < array.Length - 1) sb.Append(", ");
                }
                sb.Append("]");
                Console.WriteLine(sb.ToString());
            }
        }
    }
}
