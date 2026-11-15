using cstorchcl.core;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace cstorchcl.serialization
{
    public static class SafeTensorsSerializer
    {
        private class TensorMeta
        {
            public string dtype { get; set; } = "F32";
            public int[] shape { get; set; } = Array.Empty<int>();
            public long[] data_offsets { get; set; } = new long[2];
        }

        /// <summary>
        /// Сохраняет коллекцию тензоров в формат .safetensors
        /// </summary>
        public static unsafe void Save(string filePath, IEnumerable<Tensor> tensors, Dictionary<string, string>? metadata = null)
        {
            var tensorList = tensors.ToList();

            // Проверка на дубликаты имен (в safetensors ключи должны быть уникальными)
            var duplicateIds = tensorList.GroupBy(t => t.TensorID)
                                         .Where(g => g.Count() > 1)
                                         .Select(g => g.Key)
                                         .ToList();
            if (duplicateIds.Any())
            {
                throw new InvalidOperationException($"Обнаружены дубликаты TensorID: {string.Join(", ", duplicateIds)}");
            }

            var headerDict = new Dictionary<string, object>();
            if (metadata != null && metadata.Count > 0)
            {
                headerDict["__metadata__"] = metadata;
            }

            long currentOffset = 0;
            var rawBuffers = new List<byte[]>();
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // 1. Вычитываем данные из VRAM и формируем метаданные
            foreach (var tensor in tensorList)
            {
                long byteLength = (long)tensor.Size * sizeof(float);
                byte[] hostBytes = new byte[byteLength];

                fixed (byte* pBytes = hostBytes)
                {
                    // Синхронно читаем буфер из OpenCL в RAM
                    queue.EnqueueReadBuffer(tensor.DataBuffer, true, 0, byteLength, (IntPtr)pBytes);
                }

                rawBuffers.Add(hostBytes);

                headerDict[tensor.TensorID] = new TensorMeta
                {
                    dtype = "F32", // У вас все тензоры жестко завязаны на float (4 байта)
                    shape = tensor.Shape,
                    data_offsets = new long[] { currentOffset, currentOffset + byteLength }
                };

                currentOffset += byteLength;
            }
            queue.Finish();

            // 2. Сериализуем заголовок в UTF-8 без пробелов
            string jsonHeader = JsonSerializer.Serialize(headerDict);
            byte[] headerBytes = Encoding.UTF8.GetBytes(jsonHeader);

            // Опционально: выравнивание заголовка по 8 байт пробелами (рекомендация спецификации для mmap)
            int padding = (8 - (headerBytes.Length % 8)) % 8;
            if (padding > 0)
            {
                Array.Resize(ref headerBytes, headerBytes.Length + padding);
                for (int i = headerBytes.Length - padding; i < headerBytes.Length; i++)
                {
                    headerBytes[i] = (byte)' ';
                }
            }

            // 3. Записываем файл на диск
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            using var writer = new BinaryWriter(fs);

            // Записываем 8-байтную длину заголовка (little-endian)
            writer.Write((ulong)headerBytes.Length);
            // Записываем сам JSON-заголовок
            writer.Write(headerBytes);
            // Записываем сырые данные всех тензоров
            foreach (var buffer in rawBuffers)
            {
                writer.Write(buffer);
            }
        }
        private class TensorMetaRead
        {
            public string dtype { get; set; } = string.Empty;
            public int[] shape { get; set; } = Array.Empty<int>();
            public long[] data_offsets { get; set; } = Array.Empty<long>();
        }

        /// <summary>
        /// Загружает веса из файла .safetensors в OpenCL-буферы переданных тензоров.
        /// Возвращает словарь __metadata__, если он был в файле.
        /// </summary>
        public static unsafe Dictionary<string, string> Load(string filePath, IEnumerable<Tensor> targetTensors)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Файл весов не найден: {filePath}");

            var tensorMap = targetTensors.ToDictionary(t => t.TensorID, t => t);
            var metadata = new Dictionary<string, string>();

            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);

            // 1. Читаем длину JSON-заголовка (первые 8 байт, little-endian)
            ulong headerLength = reader.ReadUInt64();
            byte[] headerBytes = reader.ReadBytes((int)headerLength);
            string jsonHeader = Encoding.UTF8.GetString(headerBytes);

            // Смещение, с которого начинаются сырые бинарные данные тензоров
            long dataBaseOffset = 8L + (long)headerLength;

            // 2. Парсим JSON-заголовок
            using JsonDocument doc = JsonDocument.Parse(jsonHeader);
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int loadedCount = 0;

            foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "__metadata__")
                {
                    foreach (JsonProperty metaProp in prop.Value.EnumerateObject())
                    {
                        metadata[metaProp.Name] = metaProp.Value.GetString() ?? string.Empty;
                    }
                    continue;
                }

                if (!tensorMap.TryGetValue(prop.Name, out Tensor? tensor))
                {
                    // Пропускаем тензоры из файла, которых нет в текущей модели
                    continue;
                }

                var meta = JsonSerializer.Deserialize<TensorMetaRead>(prop.Value.GetRawText())
                           ?? throw new InvalidDataException($"Ошибка чтения метаданных тензора '{prop.Name}'.");

                // Валидация типа и формы тензора
                if (meta.dtype != "F32")
                    throw new NotSupportedException($"Тензор '{prop.Name}' имеет тип {meta.dtype}, ожидался F32.");

                if (!tensor.Shape.SequenceEqual(meta.shape))
                    throw new InvalidOperationException(
                        $"Несовпадение формы для '{prop.Name}': в модели [{string.Join(", ", tensor.Shape)}], " +
                        $"в файле [{string.Join(", ", meta.shape)}].");

                long startOffset = meta.data_offsets[0];
                long byteLength = meta.data_offsets[1] - startOffset;
                long expectedBytes = (long)tensor.Size * sizeof(float);

                if (byteLength != expectedBytes)
                    throw new InvalidDataException($"Неверный размер буфера для '{prop.Name}': {byteLength} != {expectedBytes} байт.");

                // 3. Читаем сырые байты из файла и отправляем во VRAM
                fs.Seek(dataBaseOffset + startOffset, SeekOrigin.Begin);
                byte[] buffer = reader.ReadBytes((int)byteLength);

                fixed (byte* pBuf = buffer)
                {
                    queue.EnqueueWriteBuffer(tensor.DataBuffer, true, 0, byteLength, (IntPtr)pBuf);
                }

                loadedCount++;
            }

            queue.Finish();

            if (loadedCount != tensorMap.Count)
            {
                var missing = tensorMap.Keys.Except(doc.RootElement.EnumerateObject().Select(p => p.Name));
                throw new InvalidOperationException($"В файле не найдены веса для тензоров: {string.Join(", ", missing)}");
            }

            return metadata;
        }
    }
}
