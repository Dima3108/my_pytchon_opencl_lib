using System;
using NOpenCL;

namespace cstorchcl.extensions
{
    public static class KernelArgumentExtensions
    {
        /// <summary>
        /// Позволяет безопасно передавать числа с плавающей точкой (float) в OpenCL-ядра.
        /// </summary>
        public static void SetValue(this KernelArgument argument, float value)
        {
            unsafe
            {
                // Передаем размер типа float (4 байта) и адрес переменной в памяти
                argument.SetValue((UIntPtr)sizeof(float), (IntPtr)(&value));
            }
        }
    }
}
