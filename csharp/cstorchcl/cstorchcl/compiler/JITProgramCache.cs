using System;
using System.Collections.Generic;
using NOpenCL;

namespace cstorchcl.compiler
{
    /// <summary>
    /// Обеспечивает кэширование и JIT-компиляцию программ OpenCL, предотвращая повторную сборку кода ядер.
    /// </summary>
    public static class JITProgramCache
    {
        private static readonly Dictionary<string, Program> _cache = new Dictionary<string, Program>();
        private static readonly object _lock = new object();

        /// <summary>
        /// Возвращает скомпилированную программу из кэша или компилирует её «на лету», если её там нет.
        /// </summary>
        public static Program GetOrCompile(Context context, Device device, string sourceCode)
        {
            lock (_lock)
            {
                // В качестве ключа используем хэш-код исходного текста (или сам текст)
                if (_cache.TryGetValue(sourceCode, out var program))
                {
                    return program;
                }

                try
                {
                    // Создаем и компилируем программу под наше устройство
                    var newProgram = context.CreateProgramWithSource(sourceCode);
                    newProgram.Build(new[] { device });

                    _cache[sourceCode] = newProgram;
                    return newProgram;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException("Ошибка JIT-компиляции OpenCL кода ядра:\n" + ex.Message, ex);
                }
            }
        }
    }
}

