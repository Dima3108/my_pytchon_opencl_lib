using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

namespace TestLSTMRealTrain
{
    /// <summary>
    /// Высокопроизводительный загрузчик датасета с проекцией файла в память (MMF),
    /// двойной буферизацией (batchIndex & 1) и поддержкой Zero-Copy (UseHostPtr).
    /// </summary>
    public unsafe class StreamDataLoader : IDisposable
    {
        private readonly int _batchSize;
        private readonly int _blockTokens;
        private readonly int _elementBytes;
        private readonly long _fileSize;
        private readonly long _subBufferBytes;
        private readonly bool _isUnifiedMemory;

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private byte* _mappedBasePtr;

        // Двойной набор указателей хоста и буферов OpenCL (слоты 0 и 1)
        private readonly IntPtr[] _encHostPtrs = new IntPtr[2];
        private readonly IntPtr[] _decHostPtrs = new IntPtr[2];
        private readonly NOpenCL.Buffer[] _encClBuffers = new NOpenCL.Buffer[2];
        private readonly NOpenCL.Buffer[] _decClBuffers = new NOpenCL.Buffer[2];

        public int TotalBatches { get; private set; }

        public StreamDataLoader(string filePath, int batchSize, int blockTokens, Context openClContext, Device device)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Файл датасета не найден: {filePath}");

            _batchSize = batchSize;
            _blockTokens = blockTokens;

            int elementTokens = _blockTokens * 2;
            _elementBytes = elementTokens * sizeof(int);
            _subBufferBytes = (long)sizeof(int) * _blockTokens * _batchSize;

            _fileSize = new FileInfo(filePath).Length;
            long totalElements = _fileSize / _elementBytes;
            TotalBatches = (int)(totalElements / _batchSize);

            // Определяем, является ли GPU интегрированным (общая память с CPU, как Intel Xe)
            _isUnifiedMemory = device.HostUnifiedMemory;

            // Проецируем бинарный файл напрямую в адресное пространство процесса
            _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _accessor = _mmf.CreateViewAccessor(0, _fileSize, MemoryMappedFileAccess.Read);
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _mappedBasePtr);

            // Инициализируем 2 слота (для четных и нечетных батчей)
            for (int slot = 0; slot < 2; slot++)
            {
                _encHostPtrs[slot] = HostPtrBufferManager.GetOrCreateBuffer($"loader_enc_slot_{slot}", (int)_subBufferBytes);
                _decHostPtrs[slot] = HostPtrBufferManager.GetOrCreateBuffer($"loader_dec_slot_{slot}", (int)_subBufferBytes);

                if (_isUnifiedMemory)
                {
                    // INTEL XE: Истинный Zero-Copy! Буфер OpenCL привязан напрямую к RAM хоста.
                    _encClBuffers[slot] = openClContext.CreateBuffer(
                        MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer,
                        _subBufferBytes,
                        _encHostPtrs[slot]
                    );
                    _decClBuffers[slot] = openClContext.CreateBuffer(
                        MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer,
                        _subBufferBytes,
                        _decHostPtrs[slot]
                    );
                }
                else
                {
                    // RTX 2060 (дискретный GPU): Буферы в быстрой GDDR6 VRAM для асинхронной DMA-записи.
                    _encClBuffers[slot] = openClContext.CreateBuffer(MemoryFlags.ReadOnly, _subBufferBytes);
                    _decClBuffers[slot] = openClContext.CreateBuffer(MemoryFlags.ReadOnly, _subBufferBytes);
                }
            }
        }

        /// <summary>
        /// Готовит батч в свободном слоте (batchIndex & 1) и возвращает готовые буферы OpenCL.
        /// </summary>
        public (NOpenCL.Buffer EncBuffer, NOpenCL.Buffer DecBuffer) GetBatch(int batchIndex)
        {
            if (batchIndex >= TotalBatches)
                batchIndex %= TotalBatches;

            // Чередуем слоты 0 и 1, чтобы не перезаписать память, которую сейчас читает GPU
            int slot = batchIndex & 1;

            long byteOffset = (long)batchIndex * _batchSize * _elementBytes;
            int* pSource = (int*)(_mappedBasePtr + byteOffset);

            int* pEnc = (int*)_encHostPtrs[slot];
            int* pDec = (int*)_decHostPtrs[slot];

            long halfElementBytes = (long)_blockTokens * sizeof(int);
            int tokensPerElement = _blockTokens * 2;

            // Раскладываем данные из MMF в выровненные буферы слота
            for (int b = 0; b < _batchSize; b++)
            {
                int elementOffset = b * tokensPerElement;
                System.Buffer.MemoryCopy(&pSource[elementOffset], &pEnc[b * _blockTokens], halfElementBytes, halfElementBytes);
                System.Buffer.MemoryCopy(&pSource[elementOffset + _blockTokens], &pDec[b * _blockTokens], halfElementBytes, halfElementBytes);
            }

            // На Intel Xe (_isUnifiedMemory == true) данные УЖЕ видны видеокарте — ничего копировать не нужно!
            // На RTX 2060 запускаем неблокирующую (false) DMA-передачу в VRAM.
            if (!_isUnifiedMemory)
            {
                CommandQueue queue = DeviceQueueRegistry.GetCurrent();
                queue.EnqueueWriteBuffer(_encClBuffers[slot], false, 0, _subBufferBytes, _encHostPtrs[slot]);
                queue.EnqueueWriteBuffer(_decClBuffers[slot], false, 0, _subBufferBytes, _decHostPtrs[slot]);
            }

            return (_encClBuffers[slot], _decClBuffers[slot]);
        }

        public void Dispose()
        {
            for (int slot = 0; slot < 2; slot++)
            {
                _encClBuffers[slot]?.Dispose();
                _decClBuffers[slot]?.Dispose();
            }

            if (_mappedBasePtr != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _mappedBasePtr = null;
            }
            _accessor?.Dispose();
            _mmf?.Dispose();
        }
    }
}
