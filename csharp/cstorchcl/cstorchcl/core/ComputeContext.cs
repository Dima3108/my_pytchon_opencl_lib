using System;
using System.Linq;
using NOpenCL;
using NOpenCL.Extensions;

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
        /// <summary>
        /// Размер варпа
        /// </summary>
        public int warp_size { get; private set; }
        /// <summary>
        /// Максимальный размер рабочей группы (X*Y*Z)
        /// </summary>
        public int MAX_WORK_GROUP_SIZE { get; private set; }
        public int MAX_WORK_GPOUP_SIZE_X { get; private set; }
        public int MAX_WORK_GPOUP_SIZE_Y { get; private set; }
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
            #region DeviceParametrs
 warp_size = (int)Device.GetWarpSizeNv();
            MAX_WORK_GROUP_SIZE = (int)Device.MaxWorkGroupSize;
            if (MAX_WORK_GROUP_SIZE < warp_size)
            {
                if (MAX_WORK_GROUP_SIZE < 4)
                {
                    MAX_WORK_GPOUP_SIZE_X = 1;
                    MAX_WORK_GPOUP_SIZE_Y = 1;
                }
                else
                {
                    if (MAX_WORK_GROUP_SIZE < 16)
                    {
                        MAX_WORK_GPOUP_SIZE_X = 2;
                        MAX_WORK_GPOUP_SIZE_Y = 2;
                    }
                    else
                    {
                        if (MAX_WORK_GROUP_SIZE < 32)
                        {
                            MAX_WORK_GPOUP_SIZE_X = 4;
                            MAX_WORK_GPOUP_SIZE_Y = 4;
                        }
                        else
                        {
                            MAX_WORK_GPOUP_SIZE_X = 8;
                            MAX_WORK_GPOUP_SIZE_Y = 4;
                        }
                    }
                }
            }
            else
            {
                MAX_WORK_GPOUP_SIZE_Y = warp_size / 8;
                MAX_WORK_GPOUP_SIZE_X = MAX_WORK_GROUP_SIZE / MAX_WORK_GPOUP_SIZE_Y;
            }
#if DEBUG
            Console.WriteLine($"максимальные размеры локальной группы для устройства на основе варпа:X={MAX_WORK_GPOUP_SIZE_X},Y={MAX_WORK_GPOUP_SIZE_Y}");
#endif
#endregion


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
