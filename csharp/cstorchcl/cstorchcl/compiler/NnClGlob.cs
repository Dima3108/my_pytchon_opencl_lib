using System;
using System.IO;
using NOpenCL;
using cstorchcl.core;

namespace cstorchcl.compiler
{
    /// <summary>
    /// Диспетчер тяжелых глобальных операций библиотеки (Softmax, RMSNorm, BMM, Permute, CausalConv).
    /// Загружает, компилирует и кэширует Си-ядра OpenCL.
    /// </summary>
    public class NnClGlob
    {
        private readonly ComputeContext _computeCtx;

        // Ссылки на скомпилированные программы
        private Program _prgRmsNorm;
        private Program _prgSoftmax;
        private Program _prgMatMul;
        private Program _prgPermute;
        private Program _prgConv;
        private Program _prgEmbedding;
        private Program _prgLstm;
        private Program _prCrEtr;//cross entropu

        // Кэшированные скомпилированные ядра (экземпляры Kernel)
        private Kernel _kernelSoftmaxForward;
        private Kernel _kernelSoftmaxBackward;
        //lstm
        private Kernel _kernelLstmForward, _kernelLstmBackward, _kernelLstmHidForward, _kernelLstmHidBackward, _kernelPadSlicedGradient;
        private Kernel _kernelOneHotShift;
        //cross entropy
        private Kernel _crossEtnropyForward, _crossEntropyBackward;
        /// <summary>
        /// Инициализирует диспетчер ядер, привязываясь к активному ComputeContext.
        /// </summary>
        public NnClGlob(ComputeContext computeCtx)
        {
            _computeCtx = computeCtx ?? throw new ArgumentNullException(nameof(computeCtx));
            LoadKernels();
        }

        /// <summary>
        /// Считывает исходный Си-код ядер из папки cl_kernels и компилирует его.
        /// </summary>
        private void LoadKernels()
        {
            // Корневой путь к файлам ядер (предполагаем структуру проекта: папка cl_kernels лежит рядом с бинарником или в корне)
            string kernelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cl_kernels");

            // Если запускаемся из среды разработки, поднимаемся выше до папки проекта
            if (!Directory.Exists(kernelsDir))
            {
                kernelsDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "cl_kernels"));
            }

            if (!Directory.Exists(kernelsDir))
                throw new DirectoryNotFoundException($"Директория с OpenCL ядрами не найдена по пути: {kernelsDir}");

            var ctx = _computeCtx.Context;
            var dev = _computeCtx.Device;

            // 1. Загрузка и компиляция Softmax
            string softmaxCode = File.ReadAllText(Path.Combine(kernelsDir, "Softmax.cl"));
            _prgSoftmax = JITProgramCache.GetOrCompile(ctx, dev, softmaxCode);

            // Сразу кэшируем сами функции-ядра для вызова
            _kernelSoftmaxForward = _prgSoftmax.CreateKernel("softmax_forward");
            _kernelSoftmaxBackward = _prgSoftmax.CreateKernel("softmax_backward");

            // 2. Загрузка RMSNorm
            string rmsnormCode = File.ReadAllText(Path.Combine(kernelsDir, "rmsnorm.cl"));
            _prgRmsNorm = JITProgramCache.GetOrCompile(ctx, dev, rmsnormCode);
            _kernelRmsNormForward = _prgRmsNorm.CreateKernel("rmsnorm_forward");
            _kernelRmsNormBackward = _prgRmsNorm.CreateKernel("rmsnorm_backward");
            // 3. Загрузка Matmul / BMM
            string matmulCode = File.ReadAllText(Path.Combine(kernelsDir, "matmul.cl"));
            _prgMatMul = JITProgramCache.GetOrCompile(ctx, dev, matmulCode);
            _kernelBmmTiled = _prgMatMul.CreateKernel("bmm_tiled");
            // 4. Загрузка Permute
            string permuteCode = File.ReadAllText(Path.Combine(kernelsDir, "permute.cl"));
            _prgPermute = JITProgramCache.GetOrCompile(ctx, dev, permuteCode);
            _kernelPermute = _prgPermute.CreateKernel("tensor_permute");
            
            // 5. Загрузка CausalConv1d
            string convCode = File.ReadAllText(Path.Combine(kernelsDir, "CausalConv1d.cl"));
            _prgConv = JITProgramCache.GetOrCompile(ctx, dev, convCode);
            _kernelConvForward = _prgConv.CreateKernel("causal_conv1d_depthwise");
            _kernelConvBackwardInput = _prgConv.CreateKernel("causal_conv1d_backward_input");
            _kernelConvBackwardWeights = _prgConv.CreateKernel("causal_conv1d_backward_weights");
            // 6. Загрузка Embedding
            string embeddingCode = File.ReadAllText(Path.Combine(kernelsDir, "embedding.cl"));
            _prgEmbedding = JITProgramCache.GetOrCompile(ctx, dev, embeddingCode);
            _kernelEmbeddingForward = _prgEmbedding.CreateKernel("embedding_forward");
            _kernelEmbeddingBackward = _prgEmbedding.CreateKernel("embedding_backward");
            //7.Загрузка LSTM:
            string lstmCode = File.ReadAllText(Path.Combine(kernelsDir, "lstm.cl"));
            _prgLstm = JITProgramCache.GetOrCompile(ctx, dev, lstmCode);
            _kernelLstmForward = _prgLstm.CreateKernel("lstm_forward");
            _kernelLstmBackward = _prgLstm.CreateKernel("lstm_backward");
            _kernelLstmHidForward = _prgLstm.CreateKernel("lstm_hid_forward");
            _kernelLstmHidBackward = _prgLstm.CreateKernel("lstm_hid_backward");
            _kernelPadSlicedGradient = _prgLstm.CreateKernel("pad_sliced_gradient_kernel");
            //shift
            string shiftcode= File.ReadAllText(Path.Combine(kernelsDir, "token_one_hot_shift.cl"));
            _prgShift= JITProgramCache.GetOrCompile(ctx, dev, shiftcode);
            _kernelOneHotShift = _prgShift.CreateKernel("token_one_hot_shift");
            string concat = File.ReadAllText(Path.Combine(kernelsDir, "cat.cl"));
            _prgCat = JITProgramCache.GetOrCompile(ctx, dev, concat);
            _kernelCatForward = _prgCat.CreateKernel("cat_2tensors_forward");
            _kernelCatBackward = _prgCat.CreateKernel("cat_2tensors_backward");
            //cross entropy
            string cr = File.ReadAllText(Path.Combine(kernelsDir, "cross_entropy.cl"));
            _prCrEtr = JITProgramCache.GetOrCompile(ctx, dev, cr);
            _crossEtnropyForward = _prCrEtr.CreateKernel("cross_entropy_forward");
            _crossEntropyBackward = _prCrEtr.CreateKernel("cross_entropy_backward");

        }
        private NOpenCL.Program _prgShift, _prgCat;
        // =========================================================================
        // РЕАЛИЗАЦИЯ ОПЕРАЦИЙ (FORWARD PASS)
        // =========================================================================

        /// <summary>
        /// Глобальная безопасная операция Softmax (Forward).
        /// Рассчитана на работу со скорами внимания размерности [B * H * L, SEQ_LEN].
        /// </summary>
        public Tensor Softmax(Tensor x, int dim = -1)
        {
            // Эквивалент проверки размерности в Python: assert dim == -1 or dim == len(x.shape) - 1
            int targetDim = dim == -1 ? x.Shape.Length - 1 : dim;
            if (targetDim != x.Shape.Length - 1)
            {
                throw new NotSupportedException("Текущая реализация ядра Softmax поддерживает редукцию только по последней оси.");
            }

            int seqLen = x.Shape[x.Shape.Length - 1]; // Последняя ось
            int numRows = x.Size / seqLen;            // Вычисляем количество строк

            // Создаем выходной тензор той же формы во VRAM (RequiresGrad пока передаем как у исходного)
            Tensor y = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            // Настройка сетки потоков (Сетка идентична вашей конфигурации на Python)
            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            // Получаем глобальную активную очередь выполнения задач из нашего реестра
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // Передаем аргументы в Си-ядро по порядку индексов (0, 1, 2, 3):
            // kernel void softmax_forward(global const float* X, global float* Y, const int seq_len)
            _kernelSoftmaxForward.Arguments[0].SetValue(x.DataBuffer);
            _kernelSoftmaxForward.Arguments[1].SetValue(y.DataBuffer);
            _kernelSoftmaxForward.Arguments[2].SetValue(seqLen);

            // Запуск выполнения ядра на GPU
            activeQueue.EnqueueNDRangeKernel(_kernelSoftmaxForward, null, new[] { globalSize }, new[] { localSize });

            // Синхронизация: гарантирует завершение вычислений перед чтением
            activeQueue.Finish();

            // Если включен режим Autograd и требуется расчет градиента, привязываем контекст создателя к тензору
            if (y.RequiresGrad)
            {
                // Создаем операцию, передавая входы и выходы
                y.Creator = new SoftmaxOp(x, y, this);
            }

            return y;
        }
        /// <summary>
        /// Обратный проход глобальной операции Softmax.
        /// Использует уже вычисленный forward-выход тензора Y для экономии памяти VRAM.
        /// </summary>
        /// <param name="gradOutput">Пришедший сверху градиент dL/dY [B * H * L, SEQ_LEN]</param>
        /// <param name="y">Исходный выход прямого прохода Softmax (вероятности) [B * H * L, SEQ_LEN]</param>
        /// <returns>Выходной градиент по входу dL/dX [B * H * L, SEQ_LEN]</returns>
        public Tensor SoftmaxBackward(Tensor gradOutput, Tensor y)
        {
            int seqLen = y.Shape[y.Shape.Length - 1];
            int numRows = y.Size / seqLen;

            // Выделяем тензор для результирующего градиента по входу (по форме совпадает с Y)
            Tensor gradInput = new Tensor(y.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            // Настройка сетки потоков (идентично forward шагу)
            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            // Передаем аргументы в Си-ядро softmax_backward:
            // kernel void softmax_backward(global const float* grad_output, global const float* Y, global float* grad_input, const int seq_len)
            _kernelSoftmaxBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelSoftmaxBackward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelSoftmaxBackward.Arguments[t++].SetValue(gradInput.DataBuffer);
            _kernelSoftmaxBackward.Arguments[t++].SetValue(seqLen);

            // Запуск ядра
            activeQueue.EnqueueNDRangeKernel(_kernelSoftmaxBackward, null, new[] { globalSize }, new[] { localSize });
            activeQueue.Finish();

            return gradInput;
        }
        #region RSNORM
        // Добавьте эти переменные в начало класса NnClGlob к остальным ядрам:
        private Kernel _kernelRmsNormForward;
        private Kernel _kernelRmsNormBackward;

        // Добавьте их инициализацию в метод LoadKernels():
        // _kernelRmsNormForward = _prgRmsNorm.CreateKernel("rmsnorm_forward");
        // _kernelRmsNormBackward = _prgRmsNorm.CreateKernel("rmsnorm_backward");

        /// <summary>
        /// Глобальная операция RMSNorm (Forward).
        /// Обрабатывает тензор любой формы, выполняя нормализацию по последней оси d_model.
        /// </summary>
        public Tensor RmsNorm(Tensor x, Tensor gamma)
        {
            int dModel = x.Shape[x.Shape.Length - 1]; // Последняя ось
            int numRows = x.Size / dModel;            // Количество строк (B * L)

            // Выходной тензор
            Tensor y = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad || gamma.RequiresGrad);

            // Буфер для кэша инвертированного RMS (нужен для Backward)
            // Размер: numRows * sizeof(float)
            NOpenCL.Buffer invRmsCache = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, numRows * sizeof(float));

            // Настройка сетки потоков
            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            // Передаем аргументы ядра:
            // kernel void rmsnorm_forward(global const float* X, global const float* gamma, global float* Y, global float* inv_rms_cache, const int d_model)
            _kernelRmsNormForward.Arguments[t++].SetValue(x.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(gamma.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(invRmsCache);
            _kernelRmsNormForward.Arguments[t++].SetValue(dModel);

            activeQueue.EnqueueNDRangeKernel(_kernelRmsNormForward, null, new[] { globalSize }, new[] { localSize });
            activeQueue.Finish();

            // Если обучаемся, сохраняем кэш слоя в GradientStorage
            if (y.RequiresGrad)
            {
                var buffers = new LayerBuffers();
                buffers.layerBuffers.Add("inv_rms_cache", invRmsCache);
                GradientStorage.Push(buffers);

                // Привязываем узел операции к графу
                y.Creator = new RMSNormOp(x, gamma, y, this);
            }
            else
            {
                // Если обучение не идет, кэш нам больше не нужен, сразу освобождаем VRAM
                invRmsCache.Dispose();
            }

            return y;
        }

        /// <summary>
        /// Обратный проход глобальной операции RMSNorm.
        /// Вычисляет градиенты по входу X и по весам Gamma.
        /// </summary>
        public unsafe (Tensor gradX, Tensor gradGamma) RmsNormBackward(Tensor gradOutput, Tensor x, Tensor gamma, NOpenCL.Buffer invRmsCache)
        {
            int dModel = x.Shape[x.Shape.Length - 1];
            int numRows = x.Size / dModel;

            // Выделяем выходные градиенты
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradGamma = new Tensor(gamma.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            // Промежуточный буфер градиентов по весам для каждой строки (для последующей редукции)
            long partSizeInBytes = (long)x.Size * sizeof(float);
            using NOpenCL.Buffer gradGammaPart = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, partSizeInBytes);

            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            // Передаем аргументы ядра:
            // kernel void rmsnorm_backward(global const float* grad_output, global const float* X, global const float* gamma, global const float* inv_rms_cache, global float* grad_input, global float* grad_gamma_part, const int d_model)
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(x.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gamma.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(invRmsCache);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradX.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradGammaPart);
            _kernelRmsNormBackward.Arguments[t++].SetValue(dModel);

            activeQueue.EnqueueNDRangeKernel(_kernelRmsNormBackward, null, new[] { globalSize }, new[] { localSize });
            activeQueue.Finish();

            // --- РЕДУКЦИЯ ГРАДИЕНТОВ ПО ВЕСАМ (Адаптация кода с numpy на C#) ---
            // Считываем частичные градиенты во временный массив на CPU для суммирования строк
            float[] hostParts = new float[numRows * dModel];
            fixed (float* hp = hostParts)
            {
                activeQueue.EnqueueReadBuffer(gradGammaPart, true, 0, hostParts.Length * sizeof(float), (IntPtr)hp);
            }
            activeQueue.Finish();

            // Суммируем градиенты от всех токенов/строк в единый вектор весов Gamma
            float[] finalGradGamma = new float[dModel];
            for (int r = 0; r < numRows; r++)
            {
                int rowOffset = r * dModel;
                for (int c = 0; c < dModel; c++)
                {
                    finalGradGamma[c] += hostParts[rowOffset + c];
                }
            }

            // Загружаем финальный саккумулированный вектор градиента обратно во VRAM буфер gradGamma
            fixed (float* fgg = finalGradGamma)
            {
                activeQueue.EnqueueWriteBuffer(gradGamma.DataBuffer, true, 0, finalGradGamma.Length * sizeof(float), (IntPtr)fgg);
            }
            activeQueue.Finish();

            return (gradX, gradGamma);
        }

        #endregion
        #region Permute
        // Добавьте эту переменную в начало класса NnClGlob к остальным ядрам:
        private Kernel _kernelPermute;

        // Добавьте её инициализацию в метод LoadKernels():
        // _kernelPermute = _prgPermute.CreateKernel("tensor_permute");

        /// <summary>
        /// Глобальная перестановка осей тензора (Аналог x.permute(...) или x.transpose(...)).
        /// Поддерживает тензоры любой размерности.
        /// </summary>
        public unsafe Tensor Permute(Tensor x, int[] axes)
        {
            if (axes.Length != x.Shape.Length)
                throw new ArgumentException("Количество осей перестановки должно совпадать с размерностью тензора.");

            // Вычисляем новую форму тензора на основе массива перестановки осей
            int[] dstShape = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                dstShape[i] = x.Shape[axes[i]];
            }

            Tensor y = new Tensor(dstShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            // Подготовка метаданных для Си-ядра: размеры осей во VRAM буферы int32
            int[] srcDims = x.Shape;
            int[] dstDims = dstShape;
            fixed (int* ax = axes, srd = srcDims, dstd = dstDims)
            {
                Context ctx = _computeCtx.Context;
                using NOpenCL.Buffer srcDimsBuf = ctx.CreateBuffer(MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer, srcDims.Length * sizeof(int), (IntPtr)srd);
                using NOpenCL.Buffer dstDimsBuf = ctx.CreateBuffer(MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer, dstDims.Length * sizeof(int), (IntPtr)dstd);
                using NOpenCL.Buffer axesBuf = ctx.CreateBuffer(MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer, axes.Length * sizeof(int), (IntPtr)ax);

                IntPtr globalSize = new IntPtr(x.Size);
                CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
                int t = 0;
                // Передаем аргументы в Си-ядро tensor_permute:
                // kernel void tensor_permute(global const float* src, global float* dst, global const int* src_dims, global const int* dst_dims, global const int* permute_axes, const int num_dims, const int total_elements)
                _kernelPermute.Arguments[t++].SetValue(x.DataBuffer);
                _kernelPermute.Arguments[t++].SetValue(y.DataBuffer);
                _kernelPermute.Arguments[t++].SetValue(srcDimsBuf);
                _kernelPermute.Arguments[t++].SetValue(dstDimsBuf);
                _kernelPermute.Arguments[t++].SetValue(axesBuf);
                _kernelPermute.Arguments[t++].SetValue(x.Shape.Length);
                _kernelPermute.Arguments[t++].SetValue(x.Size);

                activeQueue.EnqueueNDRangeKernel(_kernelPermute, new[] { globalSize }, null);
                activeQueue.Finish();
            }
            return y;
        }

        #endregion
        #region BMM
        // Добавьте эту переменную в начало класса NnClGlob к остальным ядрам:
        private Kernel _kernelBmmTiled;

        // Добавьте её инициализацию в метод LoadKernels():
        // _kernelBmmTiled = _prgMatMul.CreateKernel("bmm_tiled");

        /// <summary>
        /// Пакетное матричное умножение (Batch Matrix Multiplication).
        /// Ожидает строго 3D-тензоры: A [Batch, M, K], B [Batch, K, N] -> Выход [Batch, M, N].
        /// </summary>
        /// <summary>
        /// Пакетное матричное умножение с поддержкой Batch Broadcasting для весов модели.
        /// </summary>
        public Tensor Bmm(Tensor a, Tensor b)
        {
            if (a.Shape.Length != 3 || b.Shape.Length != 3)
                throw new ArgumentException("BMM ожидает строго 3D-тензоры формы [Batch, M, K].");

            // ПРАВИЛО BROADCASTING ДЛЯ БАТЧА:
            // Если батч у B равен 1, а у А больше 1, то разрешаем операцию (веса применятся ко всем примерам)
            if (a.Shape[0] != b.Shape[0] && b.Shape[0] != 1)
                throw new ArgumentException($"Несовпадение батч-размерностей: {a.Shape[0]} и {b.Shape[0]}.");

            if (a.Shape[2] != b.Shape[1])
                throw new ArgumentException($"Несовпадение внутренних осей матриц: K-ось {a.Shape[2]} != {b.Shape[1]}.");

            int batch = a.Shape[0]; // Итоговый батч всегда равен батчу данных (3)
            int m = a.Shape[1];
            int k = a.Shape[2];
            int n = b.Shape[2];

            int[] outShape = { batch, m, n };
            Tensor c = new Tensor(outShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: a.RequiresGrad || b.RequiresGrad);

            int batchStrideA = m * k;

            // ВАЖНОЕ АРХИТЕКТУРНОЕ ИСПРАВЛЕНИЕ:
            // Если у тензора B (веса) батч равен 1, то шаг смещения пакета во VRAM принудительно зануляется!
            // Ядро на GPU будет читать одну и ту же матрицу для каждого шага батча
            int batchStrideB = (b.Shape[0] == 1) ? 0 : (k * n);

            int batchStrideC = m * n;

            // Настройка 3D-сетки потоков под тайлы 16х16
            int gx = (int)Math.Ceiling(n / 16.0) * 16;
            int gy = (int)Math.Ceiling(m / 16.0) * 16;
            int gz = batch;

            IntPtr localSizeX = new IntPtr(16);
            IntPtr localSizeY = new IntPtr(16);
            IntPtr localSizeZ = new IntPtr(1);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            _kernelBmmTiled.Arguments[t++].SetValue(a.DataBuffer);
            _kernelBmmTiled.Arguments[t++].SetValue(b.DataBuffer);
            _kernelBmmTiled.Arguments[t++].SetValue(c.DataBuffer);
            _kernelBmmTiled.Arguments[t++].SetValue(m);
            _kernelBmmTiled.Arguments[t++].SetValue(k);
            _kernelBmmTiled.Arguments[t++].SetValue(n);
            _kernelBmmTiled.Arguments[t++].SetValue(batchStrideA);
            _kernelBmmTiled.Arguments[t++].SetValue(batchStrideB); // Передаем скорректированный шаг!
            _kernelBmmTiled.Arguments[t++].SetValue(batchStrideC);

            activeQueue.EnqueueNDRangeKernel(
                _kernelBmmTiled,
                null,
                new[] { new IntPtr(gx), new IntPtr(gy), new IntPtr(gz) },
                new[] { localSizeX, localSizeY, localSizeZ }
            );
            activeQueue.Finish();

            if (c.RequiresGrad)
            {
                c.Creator = new BmmOp(a, b, c, this);
            }

            return c;
        }


        #endregion
        #region CasualConv1d
        // Добавьте эти переменные в начало класса NnClGlob к остальным ядрам:
        private Kernel _kernelConvForward;
        private Kernel _kernelConvBackwardInput;
        private Kernel _kernelConvBackwardWeights;

        // Добавьте их инициализацию в метод LoadKernels():
        // _kernelConvForward = _prgConv.CreateKernel("causal_conv1d_depthwise");
        // _kernelConvBackwardInput = _prgConv.CreateKernel("causal_conv1d_backward_input");
        // _kernelConvBackwardWeights = _prgConv.CreateKernel("causal_conv1d_backward_weights");

        /// <summary>
        /// Каузальная одномерная Depthwise свёртка (Forward).
        /// Форма X: [Batch, Channels, Length]
        /// Форма Weight: [Channels, KernelSize]
        /// Форма Bias: [Channels]
        /// </summary>
        public Tensor CausalConv1d(Tensor x, Tensor weight, Tensor bias)
        {
            if (x.Shape.Length != 3)
                throw new ArgumentException("Входной тензор X должен иметь 3 измерения [Batch, Channels, Length].");

            int batch = x.Shape[0];
            int channels = x.Shape[1];
            int length = x.Shape[2];
            int kernelSize = weight.Shape[1];

            Tensor y = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite,
                requiresGrad: x.RequiresGrad || weight.RequiresGrad || bias.RequiresGrad);

            // Настройка 3D сетки потоков по осям временной последовательности, каналов и батча
            IntPtr globalSizeX = new IntPtr(length);
            IntPtr globalSizeY = new IntPtr(channels);
            IntPtr globalSizeZ = new IntPtr(batch);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            // Передаем аргументы ядра causal_conv1d_depthwise
            _kernelConvForward.Arguments[t++].SetValue(x.DataBuffer);
            _kernelConvForward.Arguments[t++].SetValue(weight.DataBuffer);
            _kernelConvForward.Arguments[t++].SetValue(bias.DataBuffer);
            _kernelConvForward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelConvForward.Arguments[t++].SetValue(batch);
            _kernelConvForward.Arguments[t++].SetValue(channels);
            _kernelConvForward.Arguments[t++].SetValue(length);
            _kernelConvForward.Arguments[t++].SetValue(kernelSize);

            activeQueue.EnqueueNDRangeKernel(
                _kernelConvForward,
                
                new[] { globalSizeX, globalSizeY, globalSizeZ },
                null
            );
            activeQueue.Finish();

            if (y.RequiresGrad)
            {
                y.Creator = new CausalConv1dOp(x, weight, bias, y, this);
            }

            return y;
        }

        /// <summary>
        /// Обратный проход каузальной свёртки. Вычисляет градиенты по входу, весам и смещению.
        /// </summary>
        public (Tensor gradX, Tensor gradWeight, Tensor gradBias) CausalConv1dBackward(Tensor gradOutput, Tensor x, Tensor weight)
        {
            int batch = x.Shape[0];
            int channels = x.Shape[1];
            int length = x.Shape[2];
            int kernelSize = weight.Shape[1];

            // Аллокация результирующих тензоров градиентов
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradWeight = new Tensor(weight.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradBias = new Tensor(new[] { channels }, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // 1. Запуск ядра расчета градиента по входу X (Сетка 3D: Length, Channels, Batch)
            IntPtr gSizeInX = new IntPtr(length);
            IntPtr gSizeInY = new IntPtr(channels);
            IntPtr gSizeInZ = new IntPtr(batch);
            int t = 0;
            _kernelConvBackwardInput.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelConvBackwardInput.Arguments[t++].SetValue(weight.DataBuffer);
            _kernelConvBackwardInput.Arguments[t++].SetValue(gradX.DataBuffer);
            _kernelConvBackwardInput.Arguments[t++].SetValue(batch);
            _kernelConvBackwardInput.Arguments[t++].SetValue(channels);
            _kernelConvBackwardInput.Arguments[t++].SetValue(length);
            _kernelConvBackwardInput.Arguments[t++].SetValue(kernelSize);

            activeQueue.EnqueueNDRangeKernel(
                _kernelConvBackwardInput,
               
                new[] { gSizeInX, gSizeInY, gSizeInZ },
                null
            );

            // 2. Запуск ядра расчета градиентов по параметрам Weight и Bias (Сетка 2D: KernelSize, Channels)
            IntPtr gSizeW_X = new IntPtr(kernelSize);
            IntPtr gSizeW_Y = new IntPtr(channels);
            t=0;
            _kernelConvBackwardWeights.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(x.DataBuffer);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(gradWeight.DataBuffer);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(gradBias.DataBuffer);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(batch);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(channels);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(length);
            _kernelConvBackwardWeights.Arguments[t++].SetValue(kernelSize);

            activeQueue.EnqueueNDRangeKernel(
                _kernelConvBackwardWeights,
                
                new[] { gSizeW_X, gSizeW_Y },
                null
            );

            // Дожидаемся окончания работы обоих ядер на GPU
            activeQueue.Finish();

            return (gradX, gradWeight, gradBias);
        }

        #endregion
        #region Embedding
        // Добавьте эти переменные в начало класса NnClGlob к остальным ядрам:
        private Kernel _kernelEmbeddingForward;
        private Kernel _kernelEmbeddingBackward;

        // Добавьте их инициализацию в метод LoadKernels():
        // _kernelEmbeddingForward = _prgEmbedding.CreateKernel("embedding_forward");
        // _kernelEmbeddingBackward = _prgEmbedding.CreateKernel("embedding_backward");

        /// <summary>
        /// Выборка эмбеддингов по ID токенов (Forward).
        /// </summary>
        /// <param name="xTokensBuf">Низкоуровневый OpenCL буфер, хранящий целочисленные ID токенов (int32).</param>
        /// <param name="weight">Тензор весов эмбеддингов формы [VocabSize, EmbeddingDim].</param>
        /// <param name="totalTokens">Общее количество токенов в батче (Batch * SeqLen).</param>
        /// <returns>Выходной тензор формы [Batch * SeqLen, EmbeddingDim] или эквивалентной многомерной формы.</returns>
        public Tensor Embedding(NOpenCL.Buffer xTokensBuf, Tensor weight, int totalTokens)
        {
            if (weight.Shape.Length != 2)
                throw new ArgumentException("Матрица весов Embedding должна быть строго 2D тензором [VocabSize, EmbeddingDim].");

            int vocabSize = weight.Shape[0];
            int embeddingDim = weight.Shape[1];

            // Создаем выходной тензор. Форма: [TotalTokens, EmbeddingDim]
            int[] outShape = { totalTokens, embeddingDim };
            Tensor y = new Tensor(outShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: weight.RequiresGrad);

            // Настройка 2D-сетки потоков: по количеству токенов и по размерности эмбеддинга
            IntPtr globalSizeX = new IntPtr(totalTokens);
            IntPtr globalSizeY = new IntPtr(embeddingDim);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            // Передаем аргументы в Си-ядро embedding_forward:
            // kernel void embedding_forward(global const int* X_tokens, global const float* weight, global float* Y_out, const int embedding_dim, const int total_tokens)
            _kernelEmbeddingForward.Arguments[t++].SetValue(xTokensBuf);
            _kernelEmbeddingForward.Arguments[t++].SetValue(weight.DataBuffer);
            _kernelEmbeddingForward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelEmbeddingForward.Arguments[t++].SetValue(embeddingDim);
            _kernelEmbeddingForward.Arguments[t++].SetValue(totalTokens);
            int lX = _computeCtx.MAX_WORK_GPOUP_SIZE_X;
            int lY = _computeCtx.MAX_WORK_GPOUP_SIZE_Y;
            while (globalSizeX % lX != 0) lX--;
            while (globalSizeY % lY != 0) lY--;
            activeQueue.EnqueueNDRangeKernel(
                 _kernelEmbeddingForward,

                 new[] { globalSizeX, globalSizeY },
                 new[] { (IntPtr)lX, (IntPtr)lY }
            );
            activeQueue.Finish();

            // Если матрица весов требует градиентов, регистрируем операцию в графе
            if (y.RequiresGrad)
            {
                y.Creator = new EmbeddingOp(xTokensBuf, weight, y, totalTokens, this);
            }

            return y;
        }

        /// <summary>
        /// Обратный проход слоя Embedding. Рассчитывает градиент по матрице весов (gradWeight).
        /// Альтернативная реализация: обнуление буфера через запись host-массива (замена отсутствующего FillBuffer).
        /// </summary>
        public unsafe Tensor EmbeddingBackward(Tensor gradOutput, NOpenCL.Buffer xTokensBuf, Tensor weight, int totalTokens)
        {
            int vocabSize = weight.Shape[0];
            int embeddingDim = weight.Shape[1];

            // Выделяем тензор для накопления градиентов по весам (форма совпадает с Weight)
            Tensor gradWeight = new Tensor(weight.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // =========================================================================
            // ИСПРАВЛЕНИЕ: ОБНУЛЕНИЕ БУФЕРА ЧЕРЕЗ ХОСТ-МАССИВ СИЛАМИ .NET 10
            // =========================================================================
            // Создаем массив нулей на CPU. В C# при выделении массива float[] 
            // среда выполнения .NET гарантирует, что он изначально полностью заполнен нулями.
            float[] zeroHostPattern = new float[gradWeight.Size];

            // Вычисляем размер буфера в байтах
            long bufferSizeInBytes = (long)gradWeight.Size * sizeof(float);
            fixed (float* zhp = zeroHostPattern)
            {
                // Синхронно записываем массив нулей во VRAM девайса, чтобы очистить старые данные
                // Аргументы: буфер, blocking_write (true), смещение, размер в байтах, массив-источник
                activeQueue.EnqueueWriteBuffer(gradWeight.DataBuffer, true, 0, bufferSizeInBytes, (IntPtr)zhp);
            }
            // =========================================================================

            // Настройка 2D-сетки потоков под конфигурацию обратного прохода
            IntPtr globalSizeX = new IntPtr(totalTokens);
            IntPtr globalSizeY = new IntPtr(embeddingDim);
            int t = 0;
            // Передаем аргументы в Си-ядро embedding_backward:
            // kernel void embedding_backward(global const float* grad_output, global const int* X_tokens, global float* grad_weight, const int embedding_dim, const int total_tokens)
            _kernelEmbeddingBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(xTokensBuf);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(gradWeight.DataBuffer);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(embeddingDim);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(totalTokens);

            // Запускаем атомарное ядро
            activeQueue.EnqueueNDRangeKernel(
                _kernelEmbeddingBackward,
                
                new[] { globalSizeX, globalSizeY },
                null
            );
            activeQueue.Finish();

            return gradWeight;
        }


        #endregion
#if DEBUG
        /// <summary>
        /// Временная отладочная операция ELU (Forward). Код ядра встроен прямо в метод.
        /// </summary>
        public Tensor Elu(Tensor x, float alpha = 1.0f)
        {
            Tensor y = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            string kernelSource = @"
                kernel void elu_forward(global const float* X, global float* Y) {
                    int id = get_global_id(0);
                    float val = X[id];
                    Y[id] = (val > 0.0f) ? val :"+ alpha.ToString().Replace(',','.')+@" * (exp(val) - 1.0f);
                }";

            // Используем наш JITProgramCache для предотвращения повторной компиляции
            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);
            using var kernel = prg.CreateKernel("elu_forward");

            kernel.Arguments[0].SetValue(x.DataBuffer);
            kernel.Arguments[1].SetValue(y.DataBuffer);
            //kernel.Arguments[2].SetValue(alpha);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            activeQueue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(x.Size) }, null);
            activeQueue.Finish();

            if (y.RequiresGrad)
            {
                y.Creator = new EluOp(x, y, alpha, this);
            }

            return y;
        }

        /// <summary>
        /// Временная отладочная операция ELU (Backward).
        /// </summary>
        public Tensor EluBackward(Tensor gradOutput, Tensor y, float alpha = 1.0f)
        {
            Tensor gradInput = new Tensor(y.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            string kernelSource = @"
                kernel void elu_backward(global const float* grad_output, global const float* Y, global float* grad_input) {
                    int id = get_global_id(0);
                    float y_val = Y[id];
                    grad_input[id] = (y_val > 0.0f) ? grad_output[id] : grad_output[id] * (y_val + "+alpha.ToString().Replace(',','.')+@");
                }";

            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);
            using var kernel = prg.CreateKernel("elu_backward");

            kernel.Arguments[0].SetValue(gradOutput.DataBuffer);
            kernel.Arguments[1].SetValue(y.DataBuffer);
            kernel.Arguments[2].SetValue(gradInput.DataBuffer);
            //kernel.Arguments[3].SetValue(alpha);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            activeQueue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(y.Size) }, null);
            activeQueue.Finish();

            return gradInput;
        }
#endif
        #region MSELoss
        /// <summary>
        /// Вычисляет MSE Loss и возвращает скаляр ошибки, регистрируя шаг в Autograd.
        /// </summary>
        public Tensor MseLoss(Tensor pred, Tensor target)
        {
            Tensor loss = new Tensor(new[] { 1 }, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: pred.RequiresGrad);

            // Для простоты посчитаем разницу прямо на GPU
            string kernelSource = @"
                kernel void mse_forward(global const float* P, global const float* T, global float* L, const int size) {
                    float sum = 0.0f;
                    for(int i = 0; i < size; i++) {
                        float diff = P[i] - T[i];
                        sum += diff * diff;
                    }
                    L[0] = sum / size;
                }";

            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);
            using var kernel = prg.CreateKernel("mse_forward");
            kernel.Arguments[0].SetValue(pred.DataBuffer);
            kernel.Arguments[1].SetValue(target.DataBuffer);
            kernel.Arguments[2].SetValue(loss.DataBuffer);
            kernel.Arguments[3].SetValue(pred.Size);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            queue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(1) }, null);
            queue.Finish();

            if (loss.RequiresGrad)
            {
                loss.Creator = new MseLossOp(pred, target, loss, this);
            }

            return loss;
        }

        /// <summary>
        /// Градиент для MSE Loss: dL/dPred = 2 * (Pred - Target) / Size
        /// </summary>
        public Tensor MseLossBackward(Tensor pred, Tensor target)
        {
            Tensor gradInput = new Tensor(pred.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            string kernelSource = @"
                kernel void mse_backward(global const float* P, global const float* T, global float* GI, const int size) {
                    int id = get_global_id(0);
                    GI[id] = 2.0f * (P[id] - T[id]) / size;
                }";

            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);
            using var kernel = prg.CreateKernel("mse_backward");
            kernel.Arguments[0].SetValue(pred.DataBuffer);
            kernel.Arguments[1].SetValue(target.DataBuffer);
            kernel.Arguments[2].SetValue(gradInput.DataBuffer);
            kernel.Arguments[3].SetValue(pred.Size);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            queue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(pred.Size) }, null);
            queue.Finish();

            return gradInput;
        }

        #endregion
        public Dictionary<string, Tensor> RunLocal(string pseudoCode, Dictionary<string, Tensor> tensors, Dictionary<string, float> constants = null)
        {
            return NnClLoc.Run(pseudoCode, tensors, _computeCtx, constants);
        }

        #region
        /// <summary>
        /// Глобальная высокооптимизированная операция LSTM (Forward).
        /// Ожидает 3D-тензор входа X [Batch, SeqLen, InFeatures].
        /// </summary>
        /// <param name="x">Входной тензор данных.</param>
        /// <param name="w">Матрица весов входа [InFeatures, 4 * HiddenSize].</param>
        /// <param name="u">Матрица весов скрытого состояния [HiddenSize, 4 * HiddenSize].</param>
        /// <param name="bias">Вектор смещений гейтов [4 * HiddenSize].</param>
        /// <param name="hiddenSize">Размерность скрытого состояния.</param>
        /// <param name="workspace">Управляющий буфер OpenCL для записи промежуточных состояний гейтов.</param>
        /// <returns>Выходной тензор скрытых состояний H [Batch, SeqLen, HiddenSize].</returns>
        public unsafe Tensor Lstm(Tensor x, Tensor w, Tensor u, Tensor bias, int hiddenSize, out NOpenCL.Buffer workspace)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];
#if false
            Console.WriteLine(batchSize);
#endif
            int[] outputShape = new int[] { batchSize, seqLen, hiddenSize };
            Tensor hOut = new Tensor(outputShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            long workspaceBytes = (long)seqLen * batchSize * 4 * hiddenSize * sizeof(float);
            workspace = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, workspaceBytes);

            // === ДИНАМИЧЕСКАЯ АЛЛОКАЦИЯ ПРОМЕЖУТОЧНЫХ БУФЕРОВ ===
            long tempBufferBytes = (long)batchSize * hiddenSize * sizeof(float);

            using NOpenCL.Buffer cCurrBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, tempBufferBytes);
            using NOpenCL.Buffer hCurrBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, tempBufferBytes);

            int s = 1 + (int)Math.Sqrt(batchSize);
            int lX = _computeCtx.MAX_WORK_GPOUP_SIZE_X;
            int lY = _computeCtx.MAX_WORK_GPOUP_SIZE_Y;
            while (batchSize% lX > 0) lX--;
            while (hiddenSize%lY > 0) lY--;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int argIdx = 0;
            _kernelLstmForward.Arguments[argIdx++].SetValue(x.DataBuffer);
            _kernelLstmForward.Arguments[argIdx++].SetValue(w.DataBuffer);
            _kernelLstmForward.Arguments[argIdx++].SetValue(u.DataBuffer);
            _kernelLstmForward.Arguments[argIdx++].SetValue(bias.DataBuffer);
            _kernelLstmForward.Arguments[argIdx++].SetValue(hOut.DataBuffer);
            _kernelLstmForward.Arguments[argIdx++].SetValue(workspace);

            // Передаем временные буферы в ядро
            _kernelLstmForward.Arguments[argIdx++].SetValue(cCurrBuf);
            _kernelLstmForward.Arguments[argIdx++].SetValue(hCurrBuf);

            _kernelLstmForward.Arguments[argIdx++].SetValue(batchSize);
            _kernelLstmForward.Arguments[argIdx++].SetValue(seqLen);
            _kernelLstmForward.Arguments[argIdx++].SetValue(inFeatures);
            _kernelLstmForward.Arguments[argIdx++].SetValue(hiddenSize);

            activeQueue.EnqueueNDRangeKernel(_kernelLstmForward, new IntPtr[] { (IntPtr)batchSize, (IntPtr)hiddenSize }, new nint[] { (IntPtr)lX, (IntPtr)lY });
            activeQueue.Finish();

            if (hOut.RequiresGrad)
            {
                hOut.Creator = new LstmOp(x, w, u, bias, hOut, workspace, this);
            }

            return hOut;
            // Блок using гарантирует автоматический вызов cCurrBuf.Dispose() и hCurrBuf.Dispose() здесь
        }

        /// <summary>
        /// Обратный проход глобальной операции LSTM (BPTT) с атомарным накоплением градиентов параметров.
        /// </summary>
        /// <param name="gradOutput">Пришедший сверху градиент по выходу dL/dH [Batch, SeqLen, HiddenSize].</param>
        /// <param name="x">Исходный входной тензор X прямого прохода.</param>
        /// <param name="w">Матрица весов входа W.</param>
        /// <param name="u">Матрица весов скрытого состояния U.</param>
        /// <param name="hOut">Вычисленный в Forward pass тензор скрытых состояний H.</param>
        /// <param name="workspace">Сохраненный низкоуровневый буфер сырых гейтов из прямого прохода.</param>
        /// <returns>Выходной градиент функции потерь по входу данных dL/dX [Batch, SeqLen, InFeatures].</returns>
        public Tensor LstmBackward(Tensor gradOutput, Tensor x, Tensor w, Tensor u, Tensor hOut, NOpenCL.Buffer workspace)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];
            int hiddenSize = hOut.Shape[2];

            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            w.InitGradient(_computeCtx.Context);
            u.InitGradient(_computeCtx.Context);

            // === ДИНАМИЧЕСКАЯ АЛЛОКАЦИЯ ПРОМЕЖУТОЧНЫХ БУФЕРОВ ДЛЯ BACKWARD ===
            long tempBufBytes = (long)batchSize * hiddenSize * sizeof(float);
            long gatesBufBytes = (long)batchSize * 4 * hiddenSize * sizeof(float); // Гейтов 4 штуки

            using NOpenCL.Buffer dhNextBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, tempBufBytes);
            using NOpenCL.Buffer dcNextBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, tempBufBytes);
            using NOpenCL.Buffer cAccumBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, tempBufBytes);
            using NOpenCL.Buffer gradGatesBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, gatesBufBytes);

            int s = 1 + (int)Math.Sqrt(batchSize);
            int lX = _computeCtx.MAX_WORK_GPOUP_SIZE_X;
            int lY = _computeCtx.MAX_WORK_GPOUP_SIZE_Y;
            while (batchSize % lX > 0) lX--;
            while (hiddenSize % lY > 0) lY--;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int argIdx = 0;
            _kernelLstmBackward.Arguments[argIdx++].SetValue(x.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(w.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(u.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(hOut.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(workspace);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(gradOutput.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(gradX.DataBuffer);

            _kernelLstmBackward.Arguments[argIdx++].SetValue(w.Grad.DataBuffer);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(u.Grad.DataBuffer);

            // Передаем выделенные буферы
            _kernelLstmBackward.Arguments[argIdx++].SetValue(dhNextBuf);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(dcNextBuf);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(cAccumBuf);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(gradGatesBuf);

            _kernelLstmBackward.Arguments[argIdx++].SetValue(batchSize);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(seqLen);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(inFeatures);
            _kernelLstmBackward.Arguments[argIdx++].SetValue(hiddenSize);

            activeQueue.EnqueueNDRangeKernel(_kernelLstmBackward, new IntPtr[] { (IntPtr)batchSize, (IntPtr)hiddenSize }, new nint[] { (IntPtr)lX, (IntPtr)lY });
            activeQueue.Finish();

            return gradX;
            // Все 4 буфера очищаются во VRAM прямо здесь
        }

        // Зарегистрируйте ядра _kernelLstmHidForward и _kernelLstmHidBackward в LoadKernels()

        public Tensor LstmWithWithState(Tensor x, Tensor w, Tensor u, Tensor bias, Tensor hPrev, Tensor cPrev,
                                             int hiddenSize, out Tensor finalH, out Tensor finalC, out NOpenCL.Buffer workspace)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];

            Tensor hOut = new Tensor(new[] { batchSize, seqLen, hiddenSize }, _computeCtx.Context, MemoryFlags.ReadWrite, x.RequiresGrad);
            finalH = new Tensor(new[] { batchSize, hiddenSize }, _computeCtx.Context, MemoryFlags.ReadWrite, x.RequiresGrad);
            finalC = new Tensor(new[] { batchSize, hiddenSize }, _computeCtx.Context, MemoryFlags.ReadWrite, x.RequiresGrad);

            long workspaceBytes = (long)seqLen * batchSize * 4 * hiddenSize * sizeof(float);
            workspace = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, workspaceBytes);

            IntPtr globalSize = new IntPtr(batchSize);
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int argIdx = 0;
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(x.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(w.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(u.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(bias.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(hPrev.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(cPrev.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(hOut.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(workspace);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(finalH.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(finalC.DataBuffer);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(batchSize);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(seqLen);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(inFeatures);
            _kernelLstmHidForward.Arguments[argIdx++].SetValue(hiddenSize);

            activeQueue.EnqueueNDRangeKernel(_kernelLstmHidForward, new[] { globalSize }, null);
            activeQueue.Finish();

            return hOut;
        }
        public (Tensor gradX, Tensor gradHPrev, Tensor gradCPrev) LstmHidBackward(
    Tensor gradOutput, Tensor x, Tensor w, Tensor u, Tensor bias, // <-- ТЕПЕРЬ ЯВНО ПРИНИМАЕМ BIAS
    Tensor hPrev, Tensor cPrev, Tensor hOut, NOpenCL.Buffer workspace)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];
            int hiddenSize = hOut.Shape[2];

            // Выделяем выходные тензоры градиентов для входов данных
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradHPrev = new Tensor(hPrev.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradCPrev = new Tensor(cPrev.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            // Лениво инициализируем буферы градиентов для обучаемых параметров модели
            w.InitGradient(_computeCtx.Context);
            u.InitGradient(_computeCtx.Context);
            bias.InitGradient(_computeCtx.Context); // <-- ТЕПЕРЬ ВСЕ КОРРЕКТНО, ОБЪЕКТ BIAS СУЩЕСТВУЕТ В КОНТЕКСТЕ!

            IntPtr globalSize = new IntPtr(batchSize);
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int argIdx = 0;
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(x.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(w.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(u.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(hPrev.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(cPrev.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(hOut.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(workspace);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(gradOutput.DataBuffer);

            // Градиенты, которые рассчитываются на текущем шаге
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(gradX.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(gradHPrev.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(gradCPrev.DataBuffer);

            // Атомарные буферы для накопления градиентов весов (пишем напрямую в .Grad)
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(w.Grad.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(u.Grad.DataBuffer);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(bias.Grad.DataBuffer); // <-- ПЕРЕДАЕМ НАСТОЯЩИЙ БУФЕР ГРАДИЕНТА BIAS В СИ-ЯДРО

            // Примитивы конфигурации
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(batchSize);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(seqLen);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(inFeatures);
            _kernelLstmHidBackward.Arguments[argIdx++].SetValue(hiddenSize);

            // Синхронный запуск ядра BPTT
            activeQueue.EnqueueNDRangeKernel(_kernelLstmHidBackward, new[] { globalSize }, null);
            activeQueue.Finish();

            return (gradX, gradHPrev, gradCPrev);
        }


        #endregion
        #region FusinCat
        /// <summary>
        /// Выполняет эффективное слияние двух тензоров по последней оси во VRAM.
        /// Исключает необходимость в тяжелом кастомном ядре за счет высокоуровневых команд OpenCL.
        /// </summary>
        public static void FuseBuffersToConcat(Tensor left, Tensor right, Tensor destination, int hiddenDim)
        {
            int batchSize = left.Shape[0];
            int seqLen = left.Shape[1];

            long rowBytes = (long)hiddenDim * sizeof(float);
            long destRowBytes = rowBytes * 2; // Строка назначения в 2 раза шире

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Итерируемся по строкам батча и последовательности
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    // Вычисляем плоские смещения для исходных тензоров
                    long srcOffset = (long)(b * seqLen + t) * rowBytes;

                    // Вычисляем смещения в результирующем широком тензоре
                    long destLeftOffset = (long)(b * seqLen + t) * destRowBytes;
                    long destRightOffset = destLeftOffset + rowBytes; // Сдвиг на ширину левой половины

                    // Копируем левую половину (Embedded Decoder)
                    queue.EnqueueCopyBuffer(left.DataBuffer, destination.DataBuffer, (IntPtr)srcOffset, (IntPtr)destLeftOffset, (IntPtr)rowBytes);

                    // Копируем правую половину (Encoder Outputs)
                    queue.EnqueueCopyBuffer(right.DataBuffer, destination.DataBuffer, (IntPtr)srcOffset, (IntPtr)destRightOffset, (IntPtr)rowBytes);
                }
            }
            queue.Finish();
        }

        /// <summary>
        /// Обратная операция: разрезает градиент широкого буфера на две исходные половины.
        /// </summary>
        public static void SplitConcatGradients(Tensor gradDestination, Tensor gradLeft, Tensor gradRight, int hiddenDim)
        {
            int batchSize = gradLeft.Shape[0];
            int seqLen = gradLeft.Shape[1];

            long rowBytes = (long)hiddenDim * sizeof(float);
            long srcRowBytes = rowBytes * 2;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    long destOffset = (long)(b * seqLen + t) * rowBytes;

                    long srcLeftOffset = (long)(b * seqLen + t) * srcRowBytes;
                    long srcRightOffset = srcLeftOffset + rowBytes;

                    // Вырезаем градиенты для левой половины
                    queue.EnqueueCopyBuffer(gradDestination.DataBuffer, gradLeft.DataBuffer, (IntPtr)srcLeftOffset, (IntPtr)destOffset, (IntPtr)rowBytes);

                    // Вырезаем градиенты для правой половины
                    queue.EnqueueCopyBuffer(gradDestination.DataBuffer, gradRight.DataBuffer, (IntPtr)srcRightOffset, (IntPtr)destOffset, (IntPtr)rowBytes);
                }
            }
            queue.Finish();
        }
        #endregion
        // Зарегистрируйте _kernelOneHotShift = prg.CreateKernel("token_one_hot_shift") в LoadKernels()

        public Tensor PrepareTrueTargets(NOpenCL.Buffer tokensBuf, int batchSize, int seqLen, int vocabSize)
        {
            // Новая временная шкала уменьшается на 1 из-за среза [:, 1:]
            int targetSeqLen = seqLen - 1;
            var targetShape =new int[] { batchSize, targetSeqLen, vocabSize };

            Tensor targetTensor = new Tensor(targetShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Двумерная сетка потоков: [Батчи, Шаги времени]
            var globalWorkSize = new IntPtr[] { (IntPtr)batchSize, (IntPtr)targetSeqLen };

            _kernelOneHotShift.Arguments[0].SetValue(tokensBuf);
            _kernelOneHotShift.Arguments[1].SetValue(targetTensor.DataBuffer);
            _kernelOneHotShift.Arguments[2].SetValue(seqLen);
            _kernelOneHotShift.Arguments[3].SetValue(vocabSize);

            queue.EnqueueNDRangeKernel(_kernelOneHotShift, globalWorkSize, null);
            queue.Finish();

            return targetTensor;
        }
        /// <summary>
        /// Физически расширяет усеченный градиент до оригинальной формы на GPU.
        /// Заполняет отсутствующий последний временной шаг нулями.
        /// </summary>
        public  void PadSlicedGradient(Tensor gradSliced, Tensor gradOriginal)
        {
            int batchSize = gradOriginal.Shape[0];
            int oldSeqLen = gradOriginal.Shape[1];
            int vocabSize = gradOriginal.Shape[2];

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Настраиваем двумерную сетку потоков: [Батчи, Полная длина исходной последовательности]
            var globalWorkSize = new IntPtr[] { (IntPtr)batchSize, (IntPtr)oldSeqLen };

            // Передаем аргументы строго по индексам сигнатуры ядра
            _kernelPadSlicedGradient.Arguments[0].SetValue(gradSliced.DataBuffer);
            _kernelPadSlicedGradient.Arguments[1].SetValue(gradOriginal.DataBuffer);
            _kernelPadSlicedGradient.Arguments[2].SetValue(oldSeqLen);
            _kernelPadSlicedGradient.Arguments[3].SetValue(vocabSize);

            // Асинхронно запускаем параллельное заполнение на видеокарте
            queue.EnqueueNDRangeKernel(_kernelPadSlicedGradient, globalWorkSize, null);
            queue.Finish();
        }
        public Tensor SliceLogitsForLoss(Tensor logits, ComputeContext _computeCtx)
        {
            int batchSize = logits.Shape[0];
            int oldSeqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];
            int newSeqLen = oldSeqLen - 1;

            Tensor slicedLogits = new Tensor(new int[] { batchSize, newSeqLen, vocabSize }, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: logits.RequiresGrad);

            long rowBytes = (long)vocabSize * sizeof(float);
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Копируем только первые (seqLen - 1) шагов времени для каждого батча
            for (int b = 0; b < batchSize; b++)
            {
                long srcOffset = (long)b * oldSeqLen * rowBytes;
                long destOffset = (long)b * newSeqLen * rowBytes;
                long bytesToCopy = (long)newSeqLen * rowBytes;

                queue.EnqueueCopyBuffer(logits.DataBuffer, slicedLogits.DataBuffer, (IntPtr)srcOffset, (IntPtr)destOffset, (IntPtr)bytesToCopy);
            }
            queue.Finish();

            // Привязываем создателя для автодифференцирования (SliceOp)
            if (slicedLogits.RequiresGrad)
            {
                // Сюда можно передать стандартную SliceOp операцию вашей библиотеки
                slicedLogits.Creator = new SliceOp(logits, slicedLogits, this);
            }

            return slicedLogits;
        }
        #region CAT
        /// <summary>
        /// Глобальная операция конкатенации двух тензоров по последней оси (dim = -1).
        /// </summary>
        public Tensor Cat(Tensor x1, Tensor x2, int dim = -1)
        {
            int targetDim = dim == -1 ? x1.Shape.Length - 1 : dim;
            if (targetDim != x1.Shape.Length - 1)
            {
                throw new NotSupportedException("Текущая реализация Cat поддерживает конкатенацию только по последней оси.");
            }

            // Проверяем, что все размерности кроме последней совпадают
            for (int i = 0; i < x1.Shape.Length - 1; i++)
            {
                if (x1.Shape[i] != x2.Shape[i])
                    throw new ArgumentException($"Размерности тензоров не совпадают на оси {i}: {x1.Shape[i]} != {x2.Shape[i]}");
            }

            int dim1 = x1.Shape[x1.Shape.Length - 1];
            int dim2 = x2.Shape[x2.Shape.Length - 1];
            int outDim = dim1 + dim2;

            // Вычисляем новую форму
            int[] outShape = (int[])x1.Shape.Clone();
            outShape[outShape.Length - 1] = outDim;

            int numRows = x1.Size / dim1; // Количество плоских строк (B * SeqLen)

            // Создаем результирующий тензор во VRAM
            bool requiresGrad = x1.RequiresGrad || x2.RequiresGrad;
            Tensor y = new Tensor(outShape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: requiresGrad);

            // Настройка сетки потоков (1 поток на одну строку)
            IntPtr localSize = new IntPtr(64);
            int globalWorkSize = ((numRows + 63) / 64) * 64;
            IntPtr globalSize = new IntPtr(globalWorkSize);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // Передаем аргументы в OpenCL ядро
            _kernelCatForward.Arguments[0].SetValue(x1.DataBuffer);
            _kernelCatForward.Arguments[1].SetValue(x2.DataBuffer);
            _kernelCatForward.Arguments[2].SetValue(y.DataBuffer);
            _kernelCatForward.Arguments[3].SetValue(numRows);
            _kernelCatForward.Arguments[4].SetValue(dim1);
            _kernelCatForward.Arguments[5].SetValue(dim2);

            activeQueue.EnqueueNDRangeKernel(_kernelCatForward, new[] { globalSize }, new[] { localSize });
            activeQueue.Finish();

            // Регистрация в Autograd графе
            if (y.RequiresGrad)
            {
                y.Creator = new CatOp(x1, x2, y, this);
            }

            return y;
        }

        /// <summary>
        /// Реализация обратного прохода ядра Cat для Autograd.
        /// </summary>
        public (Tensor gradInput1, Tensor gradInput2) CatBackward(Tensor gradOutput, Tensor x1, Tensor x2)
        {
            int dim1 = x1.Shape[x1.Shape.Length - 1];
            int dim2 = x2.Shape[x2.Shape.Length - 1];
            int numRows = x1.Size / dim1;

            // Выделяем память под градиенты исходных тензоров
            Tensor gradInput1 = new Tensor(x1.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradInput2 = new Tensor(x2.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            IntPtr localSize = new IntPtr(64);
            int globalWorkSize = ((numRows + 63) / 64) * 64;
            IntPtr globalSize = new IntPtr(globalWorkSize);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            _kernelCatBackward.Arguments[0].SetValue(gradOutput.DataBuffer);
            _kernelCatBackward.Arguments[1].SetValue(gradInput1.DataBuffer);
            _kernelCatBackward.Arguments[2].SetValue(gradInput2.DataBuffer);
            _kernelCatBackward.Arguments[3].SetValue(numRows);
            _kernelCatBackward.Arguments[4].SetValue(dim1);
            _kernelCatBackward.Arguments[5].SetValue(dim2);

            activeQueue.EnqueueNDRangeKernel(_kernelCatBackward, new[] { globalSize }, new[] { localSize });
            activeQueue.Finish();

            return (gradInput1, gradInput2);
        }
        Kernel _kernelCatBackward, _kernelCatForward;
        #endregion
        #region CrossEntropy
        /// <summary>
        /// Вычисляет стабильный CrossEntropyLoss напрямую из сырых логитов сети.
        /// </summary>
        public Tensor CrossEntropyLoss(Tensor pred, Tensor target)
        {
            // Ошибка — это всегда скаляр (1 элемент)
            Tensor loss = new Tensor(new[] { 1 }, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: pred.RequiresGrad);

            // Предварительно обнуляем буфер лосса на GPU, так как ядро будет делать в него прибавления (atomic)
            float[] zero = new float[] { 0.0f };
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            unsafe
            {
                fixed (float* pZ = zero)
                {
                    queue.EnqueueWriteBuffer(loss.DataBuffer, true, 0, sizeof(float), (IntPtr)pZ);
                }
            }

            int vocabSize = pred.Shape[pred.Shape.Length - 1]; // Последняя ось (6565)
            int numRows = pred.Size / vocabSize;             // B * (SeqLen - 1)

            // Компилируем/берем исходник ядра (forward код из секции 1)
            /*tring kernelSource = @"...[Вставьте сюда код ядра cross_entropy_forward]...";
             var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);

             using var kernel = prg.CreateKernel("cross_entropy_forward");*/
            //crossEtnropyForwa   guments[0].SetValue(pred.DataBuffer);

            /*ernel.Arguments[1].SetValue(target.DataBuffer);
             kernel.Arguments[2].SetValue(loss.DataBuffer);
             kernel.Arguments[3].SetValue(vocabSize);
             kernel.Arguments[4].SetValue(numRows);*/
            int t = 0;
            _crossEtnropyForward.Arguments[t++].SetValue(pred.DataBuffer);
            _crossEtnropyForward.Arguments[t++].SetValue(target.DataBuffer);
            _crossEtnropyForward.Arguments[t++].SetValue(loss.DataBuffer);
            _crossEtnropyForward.Arguments[t++].SetValue(vocabSize);
            _crossEtnropyForward.Arguments[t++].SetValue(numRows);

            // Сетка запуска: 1 рабочая группа (256 потоков) на каждую строку тензора логитов
            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            queue.EnqueueNDRangeKernel(_crossEtnropyForward,new[] { globalSize }, new[] { localSize });
            queue.Finish();

            if (loss.RequiresGrad)
            {
                loss.Creator = new CrossEntropyLossOp(pred, target, loss, this);
            }
            return loss;
        }

        /// <summary>
        /// Вычисляет градиент обратного прохода для CrossEntropyLoss.
        /// </summary>
        public Tensor CrossEntropyLossBackward(Tensor pred, Tensor target)
        {
            Tensor gradInput = new Tensor(pred.Shape, _computeCtx.Context, MemoryFlags.ReadWrite, requiresGrad: false);

            int vocabSize = pred.Shape[pred.Shape.Length - 1];
            int numRows = pred.Size / vocabSize;
            /*
            string kernelSource = @"...[Вставьте сюда код ядра cross_entropy_backward]...";
            var prg = JITProgramCache.GetOrCompile(_computeCtx.Context, _computeCtx.Device, kernelSource);

            using var kernel = prg.CreateKernel("cross_entropy_backward");*/
            /*ernel.Arguments[0].SetValue(pred.DataBuffer);
             kernel.Arguments[1].SetValue(target.DataBuffer);
             kernel.Arguments[2].SetValue(gradInput.DataBuffer);
             kernel.Arguments[3].SetValue(vocabSize);
             kernel.Arguments[4].SetValue(numRows);*/
            int t = 0;
            _crossEntropyBackward.Arguments[t++].SetValue(pred.DataBuffer);
            _crossEntropyBackward.Arguments[t++].SetValue(target.DataBuffer);
            _crossEntropyBackward.Arguments[t++].SetValue(gradInput.DataBuffer);
            _crossEntropyBackward.Arguments[t++].SetValue(vocabSize);
            _crossEntropyBackward.Arguments[t++].SetValue(numRows);

            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            queue.EnqueueNDRangeKernel(_crossEntropyBackward, new[] { globalSize }, new[] { localSize });
            queue.Finish();

            return gradInput;
        }
        #endregion
    }
}

