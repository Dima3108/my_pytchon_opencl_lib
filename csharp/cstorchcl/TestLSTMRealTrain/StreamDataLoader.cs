using System;
using System.Collections.Generic;
using System.Text;
 using NOpenCL;
using System.Runtime.InteropServices;
using cstorchcl.core;

namespace TestLSTMRealTrain
{
  /// <summary>
        /// Высокопроизводительный потоковый загрузчик бинарных датасетов (int32).
        /// Использует AllocCoTaskMem для исключения накладных расходов сборщика мусора.
        /// </summary>
        public class StreamDataLoader : IDisposable
        {
            private readonly string _filePath;
            private readonly int _batchSize;
            private readonly int _blockTokens;
            private readonly int _elementBytes;
            private readonly long _fileSize;

            public int TotalBatches { get; private set; }

            public StreamDataLoader(string filePath, int batchSize, int blockTokens)
            {
                _filePath = filePath;
                _batchSize = batchSize;
                _blockTokens = blockTokens;

                // Каждый логический элемент батча = блок энкодера + блок декодера
                int elementTokens = _blockTokens * 2;
                _elementBytes = elementTokens * sizeof(int);

                if (!File.Exists(_filePath))
                    throw new FileNotFoundException($"Файл датасета не найден: {_filePath}");

                _fileSize = new FileInfo(_filePath).Length;
                long totalElements = _fileSize / _elementBytes;
                TotalBatches = (int)(totalElements / _batchSize);
            nativeBuffer = Marshal.AllocCoTaskMem(batchSize * _elementBytes);
            encHostPtr = Marshal.AllocCoTaskMem(sizeof(int) * blockTokens*batchSize);
            decHostPtr= Marshal.AllocCoTaskMem(sizeof(int) * blockTokens * batchSize);
        }
        private IntPtr encHostPtr, decHostPtr, nativeBuffer;
            /// <summary>
            /// Извлекает батч данных из файла с помощью нативной памяти и возвращает неуправляемые буферы токенов.
            /// </summary>
            public void GetBatch(int batchIndex, Context openClContext,NOpenCL.Buffer encBuffer, NOpenCL.Buffer decBuffer)
            {
                // Корректируем индекс батча на случай выхода за границы датасета
                if (batchIndex >= TotalBatches)
                    batchIndex = batchIndex % TotalBatches;

                long byteOffset = (long)batchIndex * _batchSize * _elementBytes;
                int totalBatchBytes = _batchSize * _elementBytes;

                // Аллоцируем некопируемый нативный буфер на хосте
                //IntPtr nativeBuffer = Marshal.AllocCoTaskMem(totalBatchBytes);

                try
                {
                    using (var fs = new FileStream(_filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        fs.Seek(byteOffset, SeekOrigin.Begin);

                        // Безопасное чтение из потока в нативный указатель через Span
                        unsafe
                        {
                            var span = new Span<byte>((void*)nativeBuffer, totalBatchBytes);
                            int bytesRead = fs.Read(span);
                            if (bytesRead < totalBatchBytes)
                                throw new EndOfStreamException("Неожиданный конец файла при чтении батча.");
                        }
                    }

                    // Вычисляем размеры под-буферов в байтах
                    long halfElementBytes = _blockTokens * sizeof(int);
                    long totalSubBufferBytes = (long)_batchSize * halfElementBytes;

                    // Аллоцируем два буфера во VRAM
                    //NOpenCL.Buffer encVramBuffer = openClContext.CreateBuffer(MemoryFlags.ReadWrite, totalSubBufferBytes);
                   // NOpenCL.Buffer decVramBuffer = openClContext.CreateBuffer(MemoryFlags.ReadWrite, totalSubBufferBytes);

                    CommandQueue queue = DeviceQueueRegistry.GetCurrent();

                    // Разделяем плоский нативный массив на блоки энкодера и декодера
                    unsafe
                    {
                        int* pSource = (int*)nativeBuffer;

                        // Выделяем временные нативные буферы для копирования на GPU
                       // IntPtr encHostPtr = Marshal.AllocCoTaskMem((int)totalSubBufferBytes);
                       // IntPtr decHostPtr = Marshal.AllocCoTaskMem((int)totalSubBufferBytes);

                        int* pEnc = (int*)encHostPtr;
                        int* pDec = (int*)decHostPtr;

                        int tokensPerElement = _blockTokens * 2;

                        for (int b = 0; b < _batchSize; b++)
                        {
                            int elementOffset = b * tokensPerElement;

                            // Копируем токены энкодера
                            System.Buffer.MemoryCopy(&pSource[elementOffset], &pEnc[b * _blockTokens], totalSubBufferBytes, halfElementBytes);
                            // Копируем токены декодера
                            System.Buffer.MemoryCopy(&pSource[elementOffset + _blockTokens], &pDec[b * _blockTokens], totalSubBufferBytes, halfElementBytes);
                        }

                        // Синхронно отправляем отсортированные блоки во VRAM видеокарты
                        queue.EnqueueWriteBuffer(encBuffer, true, 0, totalSubBufferBytes, encHostPtr);
                        queue.EnqueueWriteBuffer(decBuffer, true, 0, totalSubBufferBytes, decHostPtr);

                       // Marshal.FreeCoTaskMem(encHostPtr);
                       // Marshal.FreeCoTaskMem(decHostPtr);
                    }

                    queue.Finish();
                   // return (encVramBuffer, decVramBuffer);
                }
                finally
                {
                   // Marshal.FreeCoTaskMem(nativeBuffer);
                }
            }

            public void Dispose() {
            Marshal.FreeCoTaskMem(nativeBuffer);
            Marshal.FreeCoTaskMem(decHostPtr);
            Marshal.FreeCoTaskMem(encHostPtr);
        }
        }
    }


