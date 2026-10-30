using System;
using System.Linq;
using NOpenCL;

namespace cstorchcl.core
{
    /// <summary>
    /// Управляет жизненным циклом основных компонентов OpenCL (Платформа, Устройство, Контекст).
    /// </summary>
    public class ComputeContext : IDisposable
    {
        public Platform Platform { get; private set; }
        public Device Device { get; private set; }
        public Context Context { get; private set; }

        private bool _isDisposed = false;

        /// <summary>
        /// Инициализирует OpenCL подсистему. 
        /// </summary>
        /// <param name="preferAccelerator">
        /// Если true — ищет любой НЕ центральный процессор (GPU, специализированные ускорители, Custom-девайсы).
        /// Если false — принудительно инициализирует CPU.
        /// </param>
        public ComputeContext(bool preferAccelerator = true)
        {
            // 1. Получаем доступные платформы (NVIDIA, AMD, Intel, ROCm и т.д.)
            var platforms = Platform.GetPlatforms();
            if (platforms == null || platforms.Length == 0)
                throw new InvalidOperationException("OpenCL платформы не найдены в системе.");

            // Берем первую доступную платформу
            Platform = platforms[0];

            Device[]? devices = null;

            if (preferAccelerator)
            {
                // Приоритетный список типов устройств (НЕ центральный процессор)
                // 1. Сначала ищем специализированные ускорители (AMD Instinct, NPU и т.д.)
                // 2. Затем стандартные графические процессоры (GPU)
                // 3. Затем кастомные аппаратные блоки вычислений
                DeviceType[] nonCpuTypes = {
                    DeviceType.Accelerator,
                    DeviceType.Gpu,
                    DeviceType.Custom
                };

                foreach (var type in nonCpuTypes)
                {
                    try
                    {
                        var foundDevices = Platform.GetDevices(type);
                        if (foundDevices != null && foundDevices.Length > 0)
                        {
                            devices = foundDevices;
                            System.Diagnostics.Debug.WriteLine($"Успешно найдено устройство типа: {type}");
                            break; // Устройство найдено, выходим из цикла приоритетов
                        }
                    }
                    catch
                    {
                        // Игнорируем ошибки для конкретного типа (например, если драйвер выкинул ошибку отсутствия девайсов этого типа)
                        continue;
                    }
                }

                // Каскадный откат: Если ни один ускоритель не найден, принудительно переключаемся на CPU
                if (devices == null || devices.Length == 0)
                {
                    System.Diagnostics.Debug.WriteLine("Предупреждение: Ускорители (GPU/Accelerator/Custom) не найдены. Откат на CPU.");
                    devices = Platform.GetDevices(DeviceType.Cpu);
                }
            }
            else
            {
                // Если пользователь явно попросил вычисления на CPU
                devices = Platform.GetDevices(DeviceType.Cpu);
            }

            // Проверяем финальный результат поиска устройств
            if (devices == null || devices.Length == 0)
                throw new InvalidOperationException("Не удалось найти ни одного подходящего OpenCL устройства (Ускоритель/GPU/CPU).");

            // Берем первое доступное устройство из найденных
            Device = devices[0];

            // 3. Создаем контекст для выбранного устройства
            try
            {
                Context = Context.Create(Device);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Не удалось создать контекст OpenCL для устройства '{Device.Name}' ({Device.Vendor}).", ex);
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            if (Context != null)
            {
                Context.Dispose();
            }
            _isDisposed = true;
        }
    }
}
