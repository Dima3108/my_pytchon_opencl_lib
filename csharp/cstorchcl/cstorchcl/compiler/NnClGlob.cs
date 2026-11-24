using cstorchcl.core;
using cstorchcl.extensions;
using NOpenCL;
using System;
using System.IO;

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
        private Program _prLinear;
        private Program _prSum;
        private Program _prBmmPermute;
        private Program _prgSCrossEntropy;
        // Кэшированные скомпилированные ядра (экземпляры Kernel)
        private Kernel _kernelSoftmaxForward;
        private Kernel _kernelSoftmaxBackward;
        //lstm
        private Kernel _kernelLstmForward, _kernelLstmBackward, _kernelLstmHidForward, _kernelLstmHidBackward, _kernelPadSlicedGradient;
        private Kernel _kernelOneHotShift;
        private Kernel _kernelLstmReduceGrads;
        private Kernel _kernelLstmGemmXw, _kernelLstmRecurrentForward;
        //cross entropy
        private Kernel _crossEtnropyForward, _crossEntropyBackward;
        //slice
        private Kernel _kernelSliceLogits3D;
        private NOpenCL.Program _prgSlice;
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
            _kernelReduceRowsSum = _prgRmsNorm.CreateKernel("reduce_rows_sum");
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
            //_kernelLstmForward = _prgLstm.CreateKernel("lstm_forward");
            //_kernelLstmBackward = _prgLstm.CreateKernel("lstm_backward");
            _kernelLstmHidForward = _prgLstm.CreateKernel("lstm_hid_forward");
            _kernelLstmHidBackward = _prgLstm.CreateKernel("lstm_hid_backward");
            _kernelPadSlicedGradient = _prgLstm.CreateKernel("pad_sliced_gradient_kernel");
            //_kernelLstmReduceGrads = _prgLstm.CreateKernel("lstm_reduce_grads");
            _kernelLstmRecurrentForward = _prgLstm.CreateKernel("lstm_recurrent_forward");
            _kernelLstmGemmXw = _prgLstm.CreateKernel("lstm_gemm_xw");
            //backward
            _kernelLstmRecurrentBackward = _prgLstm.CreateKernel("lstm_recurrent_backward");
            _kernelLstmGemmGradX = _prgLstm.CreateKernel("lstm_gemm_grad_x");
            _kernelLstmGemmGradWeights = _prgLstm.CreateKernel("lstm_gemm_grad_weights");

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
            //linear
            string codeLin = File.ReadAllText(Path.Combine(kernelsDir, "linear.cl"));
            _prLinear = JITProgramCache.GetOrCompile(ctx, dev, codeLin);
            _kernelLinearBiasBackward = _prLinear.CreateKernel("linear_bias_backward");
            _kernelLinearForwardTiled = _prLinear.CreateKernel("linear_forward_tiled");
            //sum
            string sumCode = File.ReadAllText(Path.Combine(kernelsDir, "reduce_sum_axis.cl"));
            _prSum= JITProgramCache.GetOrCompile(ctx, dev, sumCode);
            _kernelReduceSumAxis = _prSum.CreateKernel("reduce_sum_axis");
            _kernelReduceSumAxisBackward = _prSum.CreateKernel("reduce_sum_axis_backward");
            //bmmpermute
            string codeBP= File.ReadAllText(Path.Combine(kernelsDir, "bmm_permute_tiled.cl"));
            _prBmmPermute = JITProgramCache.GetOrCompile(ctx, dev, codeBP);
            _kernelBmmPermuteTiled = _prBmmPermute.CreateKernel("bmm_permute_tiled");
            string sliceCode = File.ReadAllText(Path.Combine(kernelsDir, "slice_logits_3d.cl"));
            _prgSlice = JITProgramCache.GetOrCompile(ctx, dev, sliceCode);
            _kernelSliceLogits3D = _prgSlice.CreateKernel("slice_logits_3d");
            //SCrossEntropy
            string sCrossEntropy = File.ReadAllText(Path.Combine(kernelsDir, "FusedSparseCrossEntropyLoss.cl"));
            _prgSCrossEntropy = JITProgramCache.GetOrCompile(ctx, dev, sCrossEntropy);
            _kernelSparseCeForward = _prgSCrossEntropy.CreateKernel("sparse_cross_entropy_forward");
            _kernelReduceMeanLoss = _prgSCrossEntropy.CreateKernel("reduce_mean_loss");
            _kernelSparseCeBackward = _prgSCrossEntropy.CreateKernel("sparse_cross_entropy_backward");
            string flashAtt= File.ReadAllText(Path.Combine(kernelsDir, "flash_attention.cl"));
            _prgFlashAtt = JITProgramCache.GetOrCompile(ctx, dev, flashAtt);
            _kernelFlashAttnForward = _prgFlashAtt.CreateKernel("flash_attention_forward");
            //_kernelFlashAttnBackward = _prgFlashAtt.CreateKernel("flash_attention_backward");
            _kernelFlashAttnBwdDq = _prgFlashAtt.CreateKernel("flash_attention_bwd_dq");
            _kernelFlashAttnBwdDkv = _prgFlashAtt.CreateKernel("flash_attention_bwd_dkv");
            //SwiGlu
            string codeSwGlu= File.ReadAllText(Path.Combine(kernelsDir, "swiglu.cl"));
            _prgSwGLU= JITProgramCache.GetOrCompile(ctx, dev, codeSwGlu);
            _kernelSwiGluForward = _prgSwGLU.CreateKernel("swiglu_gate_up_forward");
            _kernelSwiGluComputeDGateDup = _prgSwGLU.CreateKernel("swiglu_compute_dgate_dup");
            _kernelSwiGluBwdDx = _prgSwGLU.CreateKernel("swiglu_bwd_dx");
            _kernelSwiGluBwdDw = _prgSwGLU.CreateKernel("swiglu_bwd_dw");
            //add
            _prgAdd = JITProgramCache.GetOrCompile(ctx, dev,File.ReadAllText( Path.Combine(kernelsDir, "add.cl")));
            _kernelTensorAdd = _prgAdd.CreateKernel("tensor_add_forward");
          
        string wkvCode = File.ReadAllText(Path.Combine(kernelsDir, "wkv.cl"));
        _prgWkv = JITProgramCache.GetOrCompile(ctx, dev, wkvCode);
_kernelWkvForward = _prgWkv.CreateKernel("wkv_forward");
_kernelWkvBackward = _prgWkv.CreateKernel("wkv_backward");
            string rwkvOpsCode = File.ReadAllText(Path.Combine(kernelsDir, "rwkv_ops.cl"));
            _prgRwkvOps = JITProgramCache.GetOrCompile(ctx, dev, rwkvOpsCode);
            _kernelTokenShiftFwd = _prgRwkvOps.CreateKernel("token_shift_forward");
            _kernelTokenShiftBwd = _prgRwkvOps.CreateKernel("token_shift_backward");
            _kernelRwkvUnaryFwd = _prgRwkvOps.CreateKernel("rwkv_unary_forward");
            _kernelRwkvUnaryBwd = _prgRwkvOps.CreateKernel("rwkv_unary_backward");
            _kernelTensorMulFwd = _prgRwkvOps.CreateKernel("tensor_mul_forward");
            _kernelTensorMulBwd = _prgRwkvOps.CreateKernel("tensor_mul_backward");
            //droput
            _prgDropOut = JITProgramCache.GetOrCompile(ctx, dev,File.ReadAllText(Path.Combine(kernelsDir, "dropout.cl")));
            _kernelDropoutForward = _prgDropOut.CreateKernel("dropout_forward");
            _kernelDropoutBackward =_prgDropOut.CreateKernel("dropout_backward");
            //vision
            _visionPrg = JITProgramCache.GetOrCompile(ctx, dev, File.ReadAllText(Path.Combine(kernelsDir, "vision_ops.cl")));
            _kernelPatchifyForward = _visionPrg.CreateKernel("patchify_forward");
            _kernelPatchifyBackward = _visionPrg.CreateKernel("patchify_backward");
            _kernelAddPosForward = _visionPrg.CreateKernel("add_pos_embed_forward");
            _kernelAddPosBackwardPos = _visionPrg.CreateKernel("add_pos_embed_backward_pos");
        }
        //vision ops
        private Kernel _kernelPatchifyForward, _kernelPatchifyBackward, _kernelAddPosForward, _kernelAddPosBackwardPos;
        private NOpenCL.Program _visionPrg;
        private NOpenCL.Program _prgShift, _prgCat;
        private Kernel _kernelSparseCeForward, _kernelReduceMeanLoss, _kernelSparseCeBackward;
        Kernel _kernelReduceRowsSum;
        Program _prgFlashAtt,_prgSwGLU,_prgAdd;
        Kernel _kernelFlashAttnForward, _kernelFlashAttnBwdDq, _kernelFlashAttnBwdDkv;
        Kernel _kernelSwiGluForward, _kernelSwiGluBackwardX, _kernelSwiGluBackwardW, _kernelSwiGluComputeDGateDup, _kernelSwiGluBwdDx, _kernelSwiGluBwdDw;
        Kernel _kernelTensorAdd;
        #region MKV
        // В полях NnClGlob:
        private Program _prgWkv;
        private Kernel _kernelWkvForward;
        private Kernel _kernelWkvBackward;
        // --- В NnClGlob.cs (поля и LoadKernels) ---
        private Program _prgRwkvOps;
        private Kernel _kernelTokenShiftFwd, _kernelTokenShiftBwd;
        private Kernel _kernelRwkvUnaryFwd, _kernelRwkvUnaryBwd;
        private Kernel _kernelTensorMulFwd, _kernelTensorMulBwd;
        //dropuot
        private Kernel _kernelDropoutForward, _kernelDropoutBackward;
        NOpenCL.Program _prgDropOut;
        // В LoadKernels():

        #endregion
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
            Tensor y = new Tensor(x.Shape, _computeCtx.Context,"softmax_y", MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

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
            Tensor gradInput = new Tensor(y.Shape, _computeCtx.Context,"softmax_grad_Input", MemoryFlags.ReadWrite, requiresGrad: false);

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
            //activeQueue.Finish();

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
        public Tensor RmsNorm(Tensor x, Tensor gamma, string layer_id)
        {
            int dModel = x.Shape[x.Shape.Length - 1];
            int numRows = x.Size / dModel;
            bool reqGrad = x.RequiresGrad || gamma.RequiresGrad;

            Tensor y = new Tensor(x.Shape, _computeCtx.Context, $"{layer_id}.rmsnorm_y", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            // Берем переиспользуемый буфер из менеджера вместо CreateBuffer
            long cacheBytes = (long)numRows * sizeof(float);
            NOpenCL.Buffer invRmsCache = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"{layer_id}.inv_rms_cache", cacheBytes);

            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int t = 0;
            _kernelRmsNormForward.Arguments[t++].SetValue(x.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(gamma.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelRmsNormForward.Arguments[t++].SetValue(invRmsCache);
            _kernelRmsNormForward.Arguments[t++].SetValue(dModel);

            activeQueue.EnqueueNDRangeKernel(_kernelRmsNormForward, null, new[] { globalSize }, new[] { localSize });

            if (reqGrad)
            {
                y.Creator = new RMSNormOp(x, gamma, y, invRmsCache, this, layer_id);
            }

            return y;
        }

        /// <summary>
        /// Обратный проход глобальной операции RMSNorm.
        /// Вычисляет градиенты по входу X и по весам Gamma.
        /// </summary>
        public unsafe (Tensor gradX, Tensor gradGamma) RmsNormBackward(Tensor gradOutput, Tensor x, Tensor gamma, NOpenCL.Buffer invRmsCache,string layerId)
        {
            int dModel = x.Shape[x.Shape.Length - 1];
            int numRows = x.Size / dModel;

            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}.rms_norm_backward_grad_x", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradGamma = new Tensor(gamma.Shape, _computeCtx.Context, $"{layerId}.rms_norm_backward_grad_gamma", MemoryFlags.ReadWrite, requiresGrad: false);

            long partSizeInBytes = (long)x.Size * sizeof(float);
            using NOpenCL.Buffer gradGammaPart = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, partSizeInBytes);

            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // 1. Считаем gradX и частичные gradGammaPart по каждой строке
            int t = 0;
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(x.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gamma.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(invRmsCache);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradX.DataBuffer);
            _kernelRmsNormBackward.Arguments[t++].SetValue(gradGammaPart);
            _kernelRmsNormBackward.Arguments[t++].SetValue(dModel);

            activeQueue.EnqueueNDRangeKernel(_kernelRmsNormBackward, null, new[] { globalSize }, new[] { localSize });

            // 2. Суммируем столбцы gradGammaPart прямо на GPU без пересылки на CPU
            int redLocal = 64;
            int redGlobal = ((dModel + redLocal - 1) / redLocal) * redLocal;

            int rArg = 0;
            _kernelReduceRowsSum.Arguments[rArg++].SetValue(gradGammaPart);
            _kernelReduceRowsSum.Arguments[rArg++].SetValue(gradGamma.DataBuffer);
            _kernelReduceRowsSum.Arguments[rArg++].SetValue(numRows);
            _kernelReduceRowsSum.Arguments[rArg++].SetValue(dModel);

            activeQueue.EnqueueNDRangeKernel(_kernelReduceRowsSum,  new[] { new IntPtr(redGlobal) }, new[] { new IntPtr(redLocal) });
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

            Tensor y = new Tensor(dstShape, _computeCtx.Context,"permute_y", MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            // Подготовка метаданных для Си-ядра: размеры осей во VRAM буферы int32
            int[] srcDims = x.Shape;
            int[] dstDims = dstShape;
            fixed (int* ax = axes, srd = srcDims, dstd = dstDims)
            {
                Context ctx = _computeCtx.Context;
                using NOpenCL.Buffer srcDimsBuf = ctx.CreateBuffer(MemoryFlags.ReadOnly | MemoryFlags.UseHostPointer, srcDims.Length * sizeof(int), (IntPtr)srd);
                //NOpenCL.Buffer srcDimsBuf=OpenCLBufferManager.GetOrCreateBuffer("permeut_srcdimsbuf",)
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
               // activeQueue.Finish();
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
            Tensor c = new Tensor(outShape, _computeCtx.Context,"bmm_c", MemoryFlags.ReadWrite, requiresGrad: a.RequiresGrad || b.RequiresGrad);

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

            Tensor y = new Tensor(x.Shape, _computeCtx.Context,"casual_conv1d_y", MemoryFlags.ReadWrite,
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
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context,"causal_conv1d_backward_gradX", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradWeight = new Tensor(weight.Shape, _computeCtx.Context, "causal_conv1d_backward_gradWeights", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradBias = new Tensor(new[] { channels }, _computeCtx.Context, "causal_conv1d_backward_gradBias", MemoryFlags.ReadWrite, requiresGrad: false);

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

        public Tensor Embedding(NOpenCL.Buffer xTokensBuf, Tensor weight, int totalTokens,string layerId)
        {
            if (weight.Shape.Length != 2)
                throw new ArgumentException("Матрица весов Embedding должна быть строго 2D тензором [VocabSize, EmbeddingDim].");

            int embeddingDim = weight.Shape[1];

            int[] outShape = { totalTokens, embeddingDim };
            Tensor y = new Tensor(outShape, _computeCtx.Context, $"{layerId}.embedding_y",MemoryFlags.ReadWrite, requiresGrad: weight.RequiresGrad);

            // Ось 0 (быстрая) = embeddingDim, Ось 1 = totalTokens
            IntPtr globalSizeX = new IntPtr(embeddingDim);
            IntPtr globalSizeY = new IntPtr(totalTokens);

            int lX = _computeCtx.MAX_WORK_GPOUP_SIZE_X, lY = _computeCtx.MAX_WORK_GPOUP_SIZE_Y;
            while (embeddingDim % lX > 0) lX--;
            while (totalTokens % lY > 0) lY--;
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            _kernelEmbeddingForward.Arguments[t++].SetValue(xTokensBuf);
            _kernelEmbeddingForward.Arguments[t++].SetValue(weight.DataBuffer);
            _kernelEmbeddingForward.Arguments[t++].SetValue(y.DataBuffer);
            _kernelEmbeddingForward.Arguments[t++].SetValue(embeddingDim);
            _kernelEmbeddingForward.Arguments[t++].SetValue(totalTokens);

            // Передаем null в качестве local_work_size — драйвер выберет идеальный размер под субгруппу/варп устройства
            activeQueue.EnqueueNDRangeKernel(
                _kernelEmbeddingForward,
                new[] { globalSizeX, globalSizeY },
                new[] {(IntPtr)lX,(IntPtr)lY}
            );

            if (y.RequiresGrad)
            {
                y.Creator = new EmbeddingOp(xTokensBuf, weight, y, totalTokens, this,layerId);
            }
            return y;
        }

        public Tensor EmbeddingBackward(Tensor gradOutput, NOpenCL.Buffer xTokensBuf, Tensor weight, int totalTokens,string lauerId)
        {
            int vocabSize = weight.Shape[0];
            int embeddingDim = weight.Shape[1];

            // Больше НЕ нужно создавать float[gradWeight.Size] на CPU и делать EnqueueWriteBuffer!
            // Ядро само перезапишет каждый элемент gradWeight напрямую.
            Tensor gradWeight = new Tensor(weight.Shape, _computeCtx.Context, $"{lauerId}.embeddingweights_grad_weight",MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // Выравниваем сетку под локальную группу [64, 4] = 256 потоков (работает на любом GPU)
             int WG_X = (int)Math.Sqrt(_computeCtx.MAX_WORK_GROUP_SIZE);
             int WG_Y = (int)Math.Sqrt(_computeCtx.MAX_WORK_GROUP_SIZE);

            int globalX = ((embeddingDim + WG_X - 1) / WG_X) * WG_X;
            int globalY = ((vocabSize + WG_Y - 1) / WG_Y) * WG_Y;

            int t = 0;
            _kernelEmbeddingBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(xTokensBuf);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(gradWeight.DataBuffer);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(embeddingDim);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(totalTokens);
            _kernelEmbeddingBackward.Arguments[t++].SetValue(vocabSize);

            activeQueue.EnqueueNDRangeKernel(
                _kernelEmbeddingBackward,
                new[] { (IntPtr)globalX, (IntPtr)globalY },
                new[] { (IntPtr)WG_X, (IntPtr)WG_Y }
            );
          //  activeQueue.Finish();

            return gradWeight;
        }


        #endregion
#if DEBUG
        /// <summary>
        /// Временная отладочная операция ELU (Forward). Код ядра встроен прямо в метод.
        /// </summary>
        public Tensor Elu(Tensor x, float alpha = 1.0f)
        {
            Tensor y = new Tensor(x.Shape, _computeCtx.Context, "elu_y",MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

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
            Tensor gradInput = new Tensor(y.Shape, _computeCtx.Context, "elu_bacvkward_gradInput",MemoryFlags.ReadWrite, requiresGrad: false);

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
            Tensor loss = new Tensor(new[] { 1 }, _computeCtx.Context,"mseloss_loss" ,MemoryFlags.ReadWrite, requiresGrad: pred.RequiresGrad);

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
            Tensor gradInput = new Tensor(pred.Shape, _computeCtx.Context,"mselossbackward_gradInput" ,MemoryFlags.ReadWrite, requiresGrad: false);

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

        #region LSTM
        public unsafe Tensor Lstm(Tensor x, Tensor w, Tensor u, Tensor bias, int hiddenSize, out NOpenCL.Buffer workspace,string layerId)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];
            int totalTokens = batchSize * seqLen;
            int totalGates = 4 * hiddenSize;

            int[] outputShape = new int[] { batchSize, seqLen, hiddenSize };
            Tensor hOut = new Tensor(outputShape, _computeCtx.Context, $"lstm_hout_{layerId}",MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);
 CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            long workspaceBytes = (long)seqLen * batchSize * 5 * hiddenSize * sizeof(float);
            //  workspace = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, workspaceBytes);
            workspace = OpenCLBufferManager.GetOrCreateBuffer(activeQueue.Context, $"lstm_ws_{w.TensorID}_{layerId}", workspaceBytes);
            // Временные буферы берем из вашего OpenCLBufferManager
            long xwBytes = (long)totalTokens * totalGates * sizeof(float);
            long tempBufferBytes = (long)batchSize * hiddenSize * sizeof(float);

            NOpenCL.Buffer xwBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_fwd_xw_{layerId}", xwBytes);
            NOpenCL.Buffer cCurrBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_fwd_c_curr_{layerId}", tempBufferBytes);
            NOpenCL.Buffer hCurrBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_fwd_h_curr_{layerId}", tempBufferBytes);

           

            // =========================================================================
            // ЭТАП 1: Массовый тайловый GEMM (X * W + Bias) на тысячах рабочих групп
            // =========================================================================
            const int TILE_SIZE = 16;
            int globalX = ((totalGates + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            int globalY = ((totalTokens + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

            int argIdx1 = 0;
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(x.DataBuffer);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(w.DataBuffer);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(bias.DataBuffer);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(xwBuf);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(totalTokens);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(inFeatures);
            _kernelLstmGemmXw.Arguments[argIdx1++].SetValue(totalGates);

            const int TS = 16;
            const int TILE_DIM = 64; // TS * 4
             globalX = ((totalGates + TILE_DIM - 1) / TILE_DIM) * TS;
             globalY = ((totalTokens + TILE_DIM - 1) / TILE_DIM) * TS;

            activeQueue.EnqueueNDRangeKernel(
                _kernelLstmGemmXw,
                new IntPtr[] { (IntPtr)globalX, (IntPtr)globalY },
                new nint[] { (IntPtr)TS, (IntPtr)TS }
            );

            // =========================================================================
            // ЭТАП 2: Облегченный рекуррентный цикл по времени (H_{t-1} * U + XW)
            // =========================================================================
            int maxWg = (int)_computeCtx.MAX_WORK_GROUP_SIZE;
            int wgSizeY = Math.Min(256, maxWg);
            if (hiddenSize < wgSizeY)
            {
                wgSizeY = 32;
                while (wgSizeY < hiddenSize && wgSizeY < maxWg) wgSizeY *= 2;
            }

            int argIdx2 = 0;
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(xwBuf);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(u.DataBuffer);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(hOut.DataBuffer);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(workspace);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(cCurrBuf);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(hCurrBuf);
  long ctrlBytes = (long)batchSize * sizeof(int);
    NOpenCL.Buffer ctrlBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, "lstm_sync_controller", ctrlBytes);
            OpenCLBufferFiller.FillZero(ctrlBuf, ctrlBytes);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(ctrlBuf);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(batchSize);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(seqLen);
            _kernelLstmRecurrentForward.Arguments[argIdx2++].SetValue(hiddenSize);

             maxWg = (int)_computeCtx.MAX_WORK_GROUP_SIZE;
            int wgSizeX = Math.Min(maxWg, ((hiddenSize + 31) / 32) * 32);

            // Настройка запуска K групп на каждый батч (например, K = 4 блока по 256 потоков = 1024 потока на батч)
            const int WG_SIZE_X = 256;
            int blocksPerBatch = Math.Min(4, (hiddenSize + WG_SIZE_X - 1) / WG_SIZE_X);
            if (blocksPerBatch < 1) blocksPerBatch = 1;

             globalX = blocksPerBatch * WG_SIZE_X;
          

        

            // Передаем ctrlBuf в аргументы ядра и запускаем сетку [blocksPerBatch * 256, 1, batchSize]
            activeQueue.EnqueueNDRangeKernel(
                _kernelLstmRecurrentForward,
                new IntPtr[] { (IntPtr)globalX, (IntPtr)1, (IntPtr)batchSize },
                new nint[] { (IntPtr)WG_SIZE_X, (IntPtr)1, (IntPtr)1 }
            );


            //activeQueue.Finish();

            // Примечание: Creator привязывается в LSTM.Forward, здесь дублировать не нужно, если вызывается оттуда.
            return hOut;
        }
     
        public Tensor LstmBackward(Tensor gradOutput, Tensor x, Tensor w, Tensor u, Tensor bias, Tensor hOut, NOpenCL.Buffer workspace,string layerId)
        {
          
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];
            int hiddenSize = hOut.Shape[2];

            int totalTokens = batchSize * seqLen;
            int totalGates = 4 * hiddenSize;

            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, $"lstmbackward_gradx_{layerId}",MemoryFlags.ReadWrite, requiresGrad: false);
            w.InitGradient(_computeCtx.Context);
            u.InitGradient(_computeCtx.Context);
            bias.InitGradient(_computeCtx.Context);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            // Временные буферы из пула OpenCLBufferManager
            long tempBufBytes = (long)batchSize * hiddenSize * sizeof(float);
            long allGatesBytes = (long)totalTokens * totalGates * sizeof(float);

            NOpenCL.Buffer dhNextBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_bwd_dh_next_{layerId}", tempBufBytes);
            NOpenCL.Buffer dcNextBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_bwd_dc_next_{layerId}", tempBufBytes);
            NOpenCL.Buffer allGradGatesBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"lstm_bwd_all_grad_gates_{layerId}", allGatesBytes);

            // =========================================================================
            // ЭТАП 1: Рекуррентный BPTT (считает all_grad_gates и dh_next)
            // =========================================================================
            int maxWg = (int)_computeCtx.MAX_WORK_GROUP_SIZE;
            int wgSizeY = Math.Min(256, maxWg);
            /*if (hiddenSize < wgSizeY)
            {
                wgSizeY = 32;
                while (wgSizeY < hiddenSize && wgSizeY < maxWg) wgSizeY *= 2;
            }*/

            int a1 = 0;
          
            {
                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(u.DataBuffer);
                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(workspace);
                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(gradOutput.DataBuffer);

                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(allGradGatesBuf);
                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(dhNextBuf);
                _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(dcNextBuf);
            }
            long ctrlBytes = (long)batchSize * sizeof(int);
            NOpenCL.Buffer ctrlBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, "lstm_sync_controller", ctrlBytes);
            OpenCLBufferFiller.FillZero(ctrlBuf, ctrlBytes);
            _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(ctrlBuf);
            _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(batchSize);
            _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(seqLen);
            _kernelLstmRecurrentBackward.Arguments[a1++].SetValue(hiddenSize);
            const int WG_SIZE_X = 256;
            int blocksPerBatch = Math.Min(4, (hiddenSize + WG_SIZE_X - 1) / WG_SIZE_X);
            if (blocksPerBatch < 1) blocksPerBatch = 1;

          int  globalX = blocksPerBatch * WG_SIZE_X;
            activeQueue.EnqueueNDRangeKernel(
                _kernelLstmRecurrentBackward,
              new IntPtr[] { (IntPtr)globalX, (IntPtr)1, (IntPtr)batchSize },
                new nint[] { (IntPtr)WG_SIZE_X, (IntPtr)1, (IntPtr)1 }
            );

            // =========================================================================
            // ЭТАП 2: Тайловый GEMM для grad_X = all_grad_gates * W^T
            // =========================================================================
            const int TILE_SIZE = 16;
            int gxGlobalX = ((inFeatures + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            int gxGlobalY = ((totalTokens + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

            int a2 = 0;
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(allGradGatesBuf);
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(w.DataBuffer);
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(gradX.DataBuffer);
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(totalTokens);
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(inFeatures);
            _kernelLstmGemmGradX.Arguments[a2++].SetValue(totalGates);
            const int TS = 16;
            const int TILE_DIM = 64; // TS * 4

            // Этап 2: grad_X
             gxGlobalX = ((inFeatures + TILE_DIM - 1) / TILE_DIM) * TS;
             gxGlobalY = ((totalTokens + TILE_DIM - 1) / TILE_DIM) * TS;
            activeQueue.EnqueueNDRangeKernel(
                _kernelLstmGemmGradX,
                new IntPtr[] { (IntPtr)gxGlobalX, (IntPtr)gxGlobalY },
                new nint[] { (IntPtr)TS, (IntPtr)TS }
            );

            // =========================================================================
            // ЭТАП 3: Тайловый GEMM для grad_W, grad_U и grad_bias (без атомиков!)
            // =========================================================================
            int maxWeightRows = Math.Max(inFeatures, hiddenSize);
            int gwGlobalX = ((totalGates + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            int gwGlobalY = ((maxWeightRows + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

            int a3 = 0;
            
            {
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(x.DataBuffer);
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(hOut.DataBuffer);
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(allGradGatesBuf);
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(w.Grad.DataBuffer);
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(u.Grad.DataBuffer);
                _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(bias.Grad.DataBuffer);
            }
        
            _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(batchSize);
            _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(seqLen);
            _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(inFeatures);
            _kernelLstmGemmGradWeights.Arguments[a3++].SetValue(hiddenSize);

            // Этап 3: grad_W, grad_U и grad_bias
             maxWeightRows = Math.Max(inFeatures, hiddenSize);
             gwGlobalX = ((totalGates + TILE_DIM - 1) / TILE_DIM) * TS;
             gwGlobalY = ((maxWeightRows + TILE_DIM - 1) / TILE_DIM) * TS;

            activeQueue.EnqueueNDRangeKernel(
                _kernelLstmGemmGradWeights,
                new IntPtr[] { (IntPtr)gwGlobalX, (IntPtr)gwGlobalY },
                new nint[] { (IntPtr)TS, (IntPtr)TS }
            );

            // Один общий Finish() после постановки всего конвейера в очередь
            //activeQueue.Finish();

            return gradX;
        }
        private Kernel _kernelLstmGemmGradWeights, _kernelLstmGemmGradX, _kernelLstmRecurrentBackward;


        // Зарегистрируйте ядра _kernelLstmHidForward и _kernelLstmHidBackward в LoadKernels()

        public Tensor LstmWithWithState(Tensor x, Tensor w, Tensor u, Tensor bias, Tensor hPrev, Tensor cPrev,
                                             int hiddenSize, out Tensor finalH, out Tensor finalC, out NOpenCL.Buffer workspace)
        {
            int batchSize = x.Shape[0];
            int seqLen = x.Shape[1];
            int inFeatures = x.Shape[2];

            Tensor hOut = new Tensor(new[] { batchSize, seqLen, hiddenSize }, _computeCtx.Context,"lstmstate_hout", MemoryFlags.ReadWrite, x.RequiresGrad);
            finalH = new Tensor(new[] { batchSize, hiddenSize }, _computeCtx.Context,"lstmstate_fianlH", MemoryFlags.ReadWrite, x.RequiresGrad);
            finalC = new Tensor(new[] { batchSize, hiddenSize }, _computeCtx.Context,"lstmstate_finalc", MemoryFlags.ReadWrite, x.RequiresGrad);

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
            //activeQueue.Finish();

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
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, "lstmhidbackward_gradx",MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradHPrev = new Tensor(hPrev.Shape, _computeCtx.Context, "lstmhidbackward_gradhprev", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradCPrev = new Tensor(cPrev.Shape, _computeCtx.Context, "lstmhidbackward_gradcprev", MemoryFlags.ReadWrite, requiresGrad: false);

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
            //activeQueue.Finish();

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
            //queue.Finish();
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
            //queue.Finish();
        }
        #endregion
        // Зарегистрируйте _kernelOneHotShift = prg.CreateKernel("token_one_hot_shift") в LoadKernels()

        public Tensor PrepareTrueTargets(NOpenCL.Buffer tokensBuf, int batchSize, int seqLen, int vocabSize)
        {
            int targetSeqLen = seqLen - 1;
            var targetShape = new int[] { batchSize, targetSeqLen, vocabSize };
            Tensor targetTensor = new Tensor(targetShape, _computeCtx.Context, "PrepareTrueTargets_targettensor", MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Выравниваем быструю ось X вверх до кратного 256 и задаем явный localSize = [256, 1, 1]
            const int localX = 256;
            int globalX = ((vocabSize + localX - 1) / localX) * localX;

            var globalWorkSize = new IntPtr[] { (IntPtr)globalX, (IntPtr)targetSeqLen, (IntPtr)batchSize };
            var localWorkSize = new IntPtr[] { (IntPtr)localX, (IntPtr)1, (IntPtr)1 };

            _kernelOneHotShift.Arguments[0].SetValue(tokensBuf);
            _kernelOneHotShift.Arguments[1].SetValue(targetTensor.DataBuffer);
            _kernelOneHotShift.Arguments[2].SetValue(seqLen);
            _kernelOneHotShift.Arguments[3].SetValue(vocabSize);

            queue.EnqueueNDRangeKernel(_kernelOneHotShift, globalWorkSize, localWorkSize);
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
            int  lY = 16;
            while (oldSeqLen % lY > 0)
                lY--;
            // Передаем аргументы строго по индексам сигнатуры ядра
            _kernelPadSlicedGradient.Arguments[0].SetValue(gradSliced.DataBuffer);
            _kernelPadSlicedGradient.Arguments[1].SetValue(gradOriginal.DataBuffer);
            _kernelPadSlicedGradient.Arguments[2].SetValue(oldSeqLen);
            _kernelPadSlicedGradient.Arguments[3].SetValue(vocabSize);

            // Асинхронно запускаем параллельное заполнение на видеокарте
            queue.EnqueueNDRangeKernel(_kernelPadSlicedGradient, globalWorkSize, new IntPtr[] {(IntPtr)1,(IntPtr)lY});
            //queue.Finish();
        }
        public Tensor SliceLogitsForLoss(Tensor logits, ComputeContext _computeCtx)
        {
            int batchSize = logits.Shape[0];
            int oldSeqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];
            int newSeqLen = oldSeqLen - 1;

            Tensor slicedLogits = new Tensor(new int[] { batchSize, newSeqLen, vocabSize }, _computeCtx.Context, "SliceLogitsForLoss_slicedLogits", MemoryFlags.ReadWrite, requiresGrad: logits.RequiresGrad);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();

            // Каждый поток по оси X обрабатывает 4 float, группа = [256, 1, 1]
            int vecVocab = (vocabSize + 3) / 4;
            const int localX = 256;
            int globalX = ((vecVocab + localX - 1) / localX) * localX;

            int a = 0;
            _kernelSliceLogits3D.Arguments[a++].SetValue(logits.DataBuffer);
            _kernelSliceLogits3D.Arguments[a++].SetValue(slicedLogits.DataBuffer);
            _kernelSliceLogits3D.Arguments[a++].SetValue(oldSeqLen);
            _kernelSliceLogits3D.Arguments[a++].SetValue(newSeqLen);
            _kernelSliceLogits3D.Arguments[a++].SetValue(vocabSize);

            queue.EnqueueNDRangeKernel(
                _kernelSliceLogits3D,
                new[] { (IntPtr)globalX, (IntPtr)newSeqLen, (IntPtr)batchSize },
                new[] { (IntPtr)localX, (IntPtr)1, (IntPtr)1 }
            );

            if (slicedLogits.RequiresGrad)
            {
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
                throw new NotSupportedException("Текущая реализация Cat поддерживает конкатенацию только по последней оси.");

            for (int i = 0; i < x1.Shape.Length - 1; i++)
            {
                if (x1.Shape[i] != x2.Shape[i])
                    throw new ArgumentException($"Размерности тензоров не совпадают на оси {i}: {x1.Shape[i]} != {x2.Shape[i]}");
            }

            int dim1 = x1.Shape[x1.Shape.Length - 1];
            int dim2 = x2.Shape[x2.Shape.Length - 1];
            int outDim = dim1 + dim2;
            int maxDim = Math.Max(dim1, dim2);

            int[] outShape = (int[])x1.Shape.Clone();
            outShape[outShape.Length - 1] = outDim;

            int numRows = x1.Size / dim1;
            bool requiresGrad = x1.RequiresGrad || x2.RequiresGrad;
            Tensor y = new Tensor(outShape, _computeCtx.Context, "cat_y", MemoryFlags.ReadWrite, requiresGrad: requiresGrad);

            // 2D-сетка: ось X = max(dim1, dim2), ось Y = numRows
            const int WG_X = 64;
            const int WG_Y = 4;
            int globalX = ((maxDim + WG_X - 1) / WG_X) * WG_X;
            int globalY = ((numRows + WG_Y - 1) / WG_Y) * WG_Y;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            _kernelCatForward.Arguments[0].SetValue(x1.DataBuffer);
            _kernelCatForward.Arguments[1].SetValue(x2.DataBuffer);
            _kernelCatForward.Arguments[2].SetValue(y.DataBuffer);
            _kernelCatForward.Arguments[3].SetValue(numRows);
            _kernelCatForward.Arguments[4].SetValue(dim1);
            _kernelCatForward.Arguments[5].SetValue(dim2);

            activeQueue.EnqueueNDRangeKernel(
                _kernelCatForward,
                new[] { (IntPtr)globalX, (IntPtr)globalY },
                new[] { (IntPtr)WG_X, (IntPtr)WG_Y }
            );

            if (y.RequiresGrad)
            {
                y.Creator = new CatOp(x1, x2, y, this);
            }
            return y;
        }

        public (Tensor gradInput1, Tensor gradInput2) CatBackward(Tensor gradOutput, Tensor x1, Tensor x2)
        {
            int dim1 = x1.Shape[x1.Shape.Length - 1];
            int dim2 = x2.Shape[x2.Shape.Length - 1];
            int numRows = x1.Size / dim1;
            int maxDim = Math.Max(dim1, dim2);

            Tensor gradInput1 = new Tensor(x1.Shape, _computeCtx.Context, "catbackward_gradi1", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradInput2 = new Tensor(x2.Shape, _computeCtx.Context, "catbackward_gradi2", MemoryFlags.ReadWrite, requiresGrad: false);

            const int WG_X = 64;
            const int WG_Y = 4;
            int globalX = ((maxDim + WG_X - 1) / WG_X) * WG_X;
            int globalY = ((numRows + WG_Y - 1) / WG_Y) * WG_Y;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            _kernelCatBackward.Arguments[0].SetValue(gradOutput.DataBuffer);
            _kernelCatBackward.Arguments[1].SetValue(gradInput1.DataBuffer);
            _kernelCatBackward.Arguments[2].SetValue(gradInput2.DataBuffer);
            _kernelCatBackward.Arguments[3].SetValue(numRows);
            _kernelCatBackward.Arguments[4].SetValue(dim1);
            _kernelCatBackward.Arguments[5].SetValue(dim2);

            activeQueue.EnqueueNDRangeKernel(
                _kernelCatBackward,
                new[] { (IntPtr)globalX, (IntPtr)globalY },
                new[] { (IntPtr)WG_X, (IntPtr)WG_Y }
            );

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
            Tensor loss = new Tensor(new[] { 1 }, _computeCtx.Context, "CrossEntropyLoss_loss", MemoryFlags.ReadWrite, requiresGrad: pred.RequiresGrad);

            // Предварительно обнуляем буфер лосса на GPU, так как ядро будет делать в него прибавления (atomic)
           // float[] zero = new float[] { 0.0f };
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            unsafe
            {
              //  fixed (float* pZ = zero)
                {
                    float* pZ = (float*)HostPtrBufferManager.GetOrCreateBuffer("zero1", sizeof(float), 0);
                    queue.EnqueueWriteBuffer(loss.DataBuffer, false, 0, sizeof(float), (IntPtr)pZ);
                }
            }

            int vocabSize = pred.Shape[pred.Shape.Length - 1]; // Последняя ось (6565)
            int numRows = pred.Size / vocabSize;             // B * (SeqLen - 1)


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
            //queue.Finish();

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
            Tensor gradInput = new Tensor(pred.Shape, _computeCtx.Context, "CrossEntropyLossBackward_gradInput", MemoryFlags.ReadWrite, requiresGrad: false);

            int vocabSize = pred.Shape[pred.Shape.Length - 1];
            int numRows = pred.Size / vocabSize;

            int t = 0;
            
            {
                _crossEntropyBackward.Arguments[t++].SetValue(pred.DataBuffer);
                _crossEntropyBackward.Arguments[t++].SetValue(target.DataBuffer);
                _crossEntropyBackward.Arguments[t++].SetValue(gradInput.DataBuffer);
            }
       
            _crossEntropyBackward.Arguments[t++].SetValue(vocabSize);
            _crossEntropyBackward.Arguments[t++].SetValue(numRows);

            IntPtr localSize = new IntPtr(256);
            IntPtr globalSize = new IntPtr(numRows * 256);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            queue.EnqueueNDRangeKernel(_crossEntropyBackward, new[] { globalSize }, new[] { localSize });
           // queue.Finish();

            return gradInput;
        }
        #endregion

        #region Linear
        /// <summary>
        /// Монолитный прямой проход линейного слоя: Y = X * W + Bias
        /// </summary>
        /// <summary>
        /// Монолитный прямой проход линейного слоя с 4x4 регистровым тайлингом: Y = X * W + Bias
        /// </summary>
        public Tensor LinearForward(Tensor x, Tensor w, Tensor bias,string layerId)
        {
            int batch = x.Shape[0];
            int m = x.Shape[1];
            int k = x.Shape[2];
            int n = w.Shape[2];

            int[] outShape = { batch, m, n };
            bool reqGrad = x.RequiresGrad || w.RequiresGrad || bias.RequiresGrad;
            Tensor y = new Tensor(outShape, _computeCtx.Context, $"{layerId}.LinearForward_y", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            int batchStrideX = m * k;
            int batchStrideY = m * n;

            // Одна рабочая группа (16x16 потоков) обрабатывает блок 64x64 элементов выходной матрицы
            const int TS = 16;
            const int TILE_DIM = 64; // TS * 4

            int gx = ((n + TILE_DIM - 1) / TILE_DIM) * TS;
            int gy = ((m + TILE_DIM - 1) / TILE_DIM) * TS;
            int gz = batch;

            IntPtr localSizeX = new IntPtr(TS);
            IntPtr localSizeY = new IntPtr(TS);
            IntPtr localSizeZ = new IntPtr(1);

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            //if (_kernelLinearForwardTiled.Arguments.Count == 0)
           
                _kernelLinearForwardTiled.Arguments[0].SetValue(x.DataBuffer);
                _kernelLinearForwardTiled.Arguments[1].SetValue(w.DataBuffer);
                _kernelLinearForwardTiled.Arguments[2].SetValue(bias.DataBuffer);
                
_kernelLinearForwardTiled.Arguments[3].SetValue(y.DataBuffer);
            _kernelLinearForwardTiled.Arguments[4].SetValue(m);
            _kernelLinearForwardTiled.Arguments[5].SetValue(k);
            _kernelLinearForwardTiled.Arguments[6].SetValue(n);
            _kernelLinearForwardTiled.Arguments[7].SetValue(batchStrideX);
            _kernelLinearForwardTiled.Arguments[8].SetValue(batchStrideY);
          
                
           
           


            activeQueue.EnqueueNDRangeKernel(
                _kernelLinearForwardTiled,
                new[] { new IntPtr(gx), new IntPtr(gy), new IntPtr(gz) },
                new[] { localSizeX, localSizeY, localSizeZ }
            );

            if (y.RequiresGrad)
            {
                y.Creator = new LinearOp(x, w, bias, y, this,layerId);
            }

            return y;
        }


        /// <summary>
        /// Вычисляет градиент по вектору смещения Bias: сумма gradOutput по осям Batch и M.
        /// </summary>
        public Tensor LinearBiasBackward(Tensor gradOutput, int[] biasShape, string layerId)
        {
            int batch = gradOutput.Shape[0];
            int m = gradOutput.Shape[1];
            int n = gradOutput.Shape[2];
            int totalRows = batch * m;

            Tensor gradBias = new Tensor(biasShape, _computeCtx.Context, $"{layerId}.LinearBiasBackward_gradbias", MemoryFlags.ReadWrite, requiresGrad: false);
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            // Запускаем по 1 потоку на каждый выходной признак N с группой 256
            int localX = (int)256;
            while (n % localX > 0)
                localX /= 2;
            //int globalX = n;


            int t = 0;
            _kernelLinearBiasBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelLinearBiasBackward.Arguments[t++].SetValue(gradBias.DataBuffer);
            _kernelLinearBiasBackward.Arguments[t++].SetValue(totalRows);
            _kernelLinearBiasBackward.Arguments[t++].SetValue(n);

            activeQueue.EnqueueNDRangeKernel(
                _kernelLinearBiasBackward,
                new[] { (IntPtr)n },
               // null
                new[] { (IntPtr)localX }
            );

            return gradBias;
        }


        private Kernel _kernelLinearBiasBackward, _kernelLinearForwardTiled;
        #endregion

        #region Sum
        /// <summary>
        /// Суммирует элементы тензора вдоль заданной оси.
        /// </summary>
        /// <param name="x">Входной тензор.</param>
        /// <param name="axis">Индекс оси для редукции (поддерживает отрицательные значения, например -1).</param>
        /// <param name="keepDims">Если true, сохраняет редуцированную ось с размером 1.</param>
        public Tensor Sum(Tensor x, int axis, string layerId,bool keepDims = false)
        {
            int targetAxis = axis < 0 ? x.Shape.Length + axis : axis;
            if (targetAxis < 0 || targetAxis >= x.Shape.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(axis), $"Ось {axis} выходит за пределы размерности тензора {x.Shape.Length}.");
            }

            // 1. Вычисляем outer_size, axis_size и inner_size
            int outerSize = 1;
            for (int i = 0; i < targetAxis; i++) outerSize *= x.Shape[i];

            int axisSize = x.Shape[targetAxis];

            int innerSize = 1;
            for (int i = targetAxis + 1; i < x.Shape.Length; i++) innerSize *= x.Shape[i];

            // 2. Формируем выходную размерность (Shape)
            int[] outShape;
            if (keepDims)
            {
                outShape = (int[])x.Shape.Clone();
                outShape[targetAxis] = 1;
            }
            else
            {
                if (x.Shape.Length == 1)
                {
                    outShape = new[] { 1 }; // Суммирование 1D-вектора дает скаляр [1]
                }
                else
                {
                    outShape = new int[x.Shape.Length - 1];
                    int idx = 0;
                    for (int i = 0; i < x.Shape.Length; i++)
                    {
                        if (i != targetAxis) outShape[idx++] = x.Shape[i];
                    }
                }
            }

            Tensor y = new Tensor(outShape, _computeCtx.Context,layerId+"."+ "sum_y",MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            // 3. Настраиваем 2D-сетку потоков [outerSize, innerSize]
            //IntPtr[] globalSize = { (IntPtr)outerSize, (IntPtr)innerSize };
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
           
            int t = 0;
            _kernelReduceSumAxis.Arguments[t++].SetValue(x.DataBuffer);
            _kernelReduceSumAxis.Arguments[t++].SetValue(y.DataBuffer);
            _kernelReduceSumAxis.Arguments[t++].SetValue(outerSize);
            _kernelReduceSumAxis.Arguments[t++].SetValue(axisSize);
            _kernelReduceSumAxis.Arguments[t++].SetValue(innerSize);

            // Передаем null в качестве localWorkSize, чтобы драйвер OpenCL сам оптимально разбил сетку
            //activeQueue.EnqueueNDRangeKernel(_kernelReduceSumAxis, globalSize,loc );
            //activeQueue.Finish();
            int localX = (innerSize >= 32) ? 32 : 1;
            int localY = 256 / localX; // 8 или 256 (всегда 256 потоков в группе!)

            int globalX = ((innerSize + localX - 1) / localX) * localX;
            int globalY = outerSize * localY;

            activeQueue.EnqueueNDRangeKernel(
                _kernelReduceSumAxis,
                new[] { (IntPtr)globalX, (IntPtr)globalY },
                new[] { (IntPtr)localX, (IntPtr)localY }
            );
            if (y.RequiresGrad)
            {
                y.Creator = new SumOp(x, y, targetAxis, this);
            }

            return y;
        }

        /// <summary>
        /// Обратный проход операции Sum: транслирует градиент обратно по редуцированной оси.
        /// </summary>
        public Tensor SumBackward(Tensor gradOutput, int[] inputShape, int targetAxis)
        {
            int outerSize = 1;
            for (int i = 0; i < targetAxis; i++) outerSize *= inputShape[i];

            int axisSize = inputShape[targetAxis];

            int innerSize = 1;
            for (int i = targetAxis + 1; i < inputShape.Length; i++) innerSize *= inputShape[i];

            Tensor gradInput = new Tensor(inputShape, _computeCtx.Context,"sumbackward_gradInput", MemoryFlags.ReadWrite, requiresGrad: false);

           // IntPtr[] globalSize = { (IntPtr)outerSize, (IntPtr)innerSize };
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();

            int t = 0;
            int totalElements = outerSize * axisSize * innerSize;
            const int bwdLocal = 256;
            int bwdGlobal = ((totalElements + bwdLocal - 1) / bwdLocal) * bwdLocal;

            t = 0;
            _kernelReduceSumAxisBackward.Arguments[t++].SetValue(gradOutput.DataBuffer);
            _kernelReduceSumAxisBackward.Arguments[t++].SetValue(gradInput.DataBuffer);
            _kernelReduceSumAxisBackward.Arguments[t++].SetValue(totalElements);
            _kernelReduceSumAxisBackward.Arguments[t++].SetValue(axisSize);
            _kernelReduceSumAxisBackward.Arguments[t++].SetValue(innerSize);

            activeQueue.EnqueueNDRangeKernel(
                _kernelReduceSumAxisBackward,
                new[] { (IntPtr)bwdGlobal },
                new[] { (IntPtr)bwdLocal }
            );

            //activeQueue.EnqueueNDRangeKernel(_kernelReduceSumAxisBackward, globalSize, null);
            //activeQueue.Finish();

            return gradInput;
        }
        private Kernel _kernelReduceSumAxis, _kernelReduceSumAxisBackward;
        #endregion
        #region BmmPermute
        /// <summary>
        /// Пакетное матричное умножение с возможностью транспонирования матриц на лету (без аллокации Permute).
        /// </summary>
        public Tensor BmmPermute(Tensor a, Tensor b,string layerId, bool transA = false, bool transB = false)
        {
            if (a.Shape.Length != 3 || b.Shape.Length != 3)
                throw new ArgumentException("BmmPermute ожидает 3D-тензоры.");

            int batch = a.Shape[0];
            int m = transA ? a.Shape[2] : a.Shape[1];
            int kA = transA ? a.Shape[1] : a.Shape[2];
            int kB = transB ? b.Shape[2] : b.Shape[1];
            int n = transB ? b.Shape[1] : b.Shape[2];

            if (kA != kB)
                throw new ArgumentException($"Несовпадение внутренних осей: {kA} != {kB}.");

            int[] outShape = { batch, m, n };
            Tensor c = new Tensor(outShape, _computeCtx.Context, $"{layerId}.bmm_permute_c",MemoryFlags.ReadWrite, requiresGrad: false);

            int batchStrideA = (a.Shape[0] == 1) ? 0 : (a.Shape[1] * a.Shape[2]);
            int batchStrideB = (b.Shape[0] == 1) ? 0 : (b.Shape[1] * b.Shape[2]);
            int batchStrideC = m * n;

            // Одна рабочая группа (16x16 потоков) обрабатывает тайл 64x64
            const int TS = 16;
            const int TILE_DIM = 64; // TS * 4

            int gx = ((n + TILE_DIM - 1) / TILE_DIM) * TS;
            int gy = ((m + TILE_DIM - 1) / TILE_DIM) * TS;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int t = 0;
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(a.DataBuffer);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(b.DataBuffer);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(c.DataBuffer);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(m);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(kA);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(n);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(batchStrideA);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(batchStrideB);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(batchStrideC);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(transA ? 1 : 0);
            _kernelBmmPermuteTiled.Arguments[t++].SetValue(transB ? 1 : 0);

            activeQueue.EnqueueNDRangeKernel(
                _kernelBmmPermuteTiled,
                new[] { new IntPtr(gx), new IntPtr(gy), new IntPtr(batch) },
                new[] { new IntPtr(TS), new IntPtr(TS), new IntPtr(1) }
            );

            return c;
        }

        private Kernel _kernelBmmPermuteTiled;
        #endregion
        #region SCrossEnytropy
        public Tensor SparseCrossEntropyLoss(Tensor logits, NOpenCL.Buffer targetsBuf)
        {
            int batchSize = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];
            int targetSeqLen = seqLen - 1;
            int numValidTokens = batchSize * targetSeqLen;

            // Временный буфер для ошибок отдельных токенов (всего ~48 КБ вместо 315 МБ!)
            long tokenLossesBytes = (long)numValidTokens * sizeof(float);
            NOpenCL.Buffer tokenLossesBuf = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, "sparse_ce_token_losses", tokenLossesBytes);

            Tensor lossTensor = new Tensor(new int[] { 1 }, _computeCtx.Context, "SparseCE_Loss", MemoryFlags.ReadWrite, requiresGrad: logits.RequiresGrad);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            const int WG_SIZE = 256;

            // 1. Считаем ошибку по каждому валидному токену (b, t)
            int a1 = 0;
            _kernelSparseCeForward.Arguments[a1++].SetValue(logits.DataBuffer);
            _kernelSparseCeForward.Arguments[a1++].SetValue(targetsBuf);
            _kernelSparseCeForward.Arguments[a1++].SetValue(tokenLossesBuf);
            _kernelSparseCeForward.Arguments[a1++].SetValue(seqLen);
            _kernelSparseCeForward.Arguments[a1++].SetValue(vocabSize);

            queue.EnqueueNDRangeKernel(
                _kernelSparseCeForward,
                new[] { (IntPtr)WG_SIZE, (IntPtr)targetSeqLen, (IntPtr)batchSize },
                new[] { (IntPtr)WG_SIZE, (IntPtr)1, (IntPtr)1 }
            );

            // 2. Усредняем в скалярный Loss
            int a2 = 0;
            _kernelReduceMeanLoss.Arguments[a2++].SetValue(tokenLossesBuf);
            _kernelReduceMeanLoss.Arguments[a2++].SetValue(lossTensor.DataBuffer);
            _kernelReduceMeanLoss.Arguments[a2++].SetValue(numValidTokens);

            queue.EnqueueNDRangeKernel(
                _kernelReduceMeanLoss,
                new[] { (IntPtr)WG_SIZE },
                new[] { (IntPtr)WG_SIZE }
            );

            if (lossTensor.RequiresGrad)
            {
                lossTensor.Creator = new SparseCrossEntropyOp(logits, targetsBuf, lossTensor, this);
            }

            return lossTensor;
        }

        public Tensor SparseCrossEntropyBackward(Tensor gradLoss, Tensor logits, NOpenCL.Buffer targetsBuf)
        {
            int batchSize = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            Tensor gradLogits = new Tensor(logits.Shape, _computeCtx.Context, "SparseCE_GradLogits", MemoryFlags.ReadWrite, requiresGrad: false);

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            const int WG_SIZE = 256;

            int a = 0;
            _kernelSparseCeBackward.Arguments[a++].SetValue(logits.DataBuffer);
            _kernelSparseCeBackward.Arguments[a++].SetValue(targetsBuf);
            _kernelSparseCeBackward.Arguments[a++].SetValue(gradLoss.DataBuffer);
            _kernelSparseCeBackward.Arguments[a++].SetValue(gradLogits.DataBuffer);
            _kernelSparseCeBackward.Arguments[a++].SetValue(batchSize);
            _kernelSparseCeBackward.Arguments[a++].SetValue(seqLen);
            _kernelSparseCeBackward.Arguments[a++].SetValue(vocabSize);

            queue.EnqueueNDRangeKernel(
                _kernelSparseCeBackward,
                new[] { (IntPtr)WG_SIZE, (IntPtr)seqLen, (IntPtr)batchSize },
                new[] { (IntPtr)WG_SIZE, (IntPtr)1, (IntPtr)1 }
            );

            return gradLogits;
        }

        #endregion
        #region FlashAttenction
        /// <summary>
        /// Fused FlashAttention (Backward).
        /// </summary>
        public Tensor FlashAttention(Tensor q, Tensor k, Tensor v, int numHeads, bool isCausal = true, string layerId = "flash_attn")
        {
            int batchSize = q.Shape[0];
            int seqLen = q.Shape[1];
            int totalDim = q.Shape[2];
            int headDim = totalDim / numHeads;

            if (headDim > 128)
                throw new ArgumentException($"HeadDim ({headDim}) превышает MAX_HEAD_DIM (128).");

            const int TILE_SIZE = 16;
            float scale = 1.0f / (float)Math.Sqrt(headDim);
            bool reqGrad = q.RequiresGrad || k.RequiresGrad || v.RequiresGrad;

            Tensor output = new Tensor(q.Shape, _computeCtx.Context, $"{layerId}_out", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            int numQTiles = (seqLen + TILE_SIZE - 1) / TILE_SIZE;
            IntPtr[] localSize = new[] { new IntPtr(TILE_SIZE), new IntPtr(1) };
            IntPtr[] globalSize = new[] { new IntPtr(numQTiles * TILE_SIZE), new IntPtr(batchSize * numHeads) };

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelFlashAttnForward.Arguments[arg++].SetValue(q.DataBuffer);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(k.DataBuffer);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(v.DataBuffer);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(output.DataBuffer);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(seqLen);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(numHeads);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(headDim);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(scale);
            _kernelFlashAttnForward.Arguments[arg++].SetValue(isCausal ? 1 : 0);

            activeQueue.EnqueueNDRangeKernel(_kernelFlashAttnForward, null, globalSize, localSize);
            //activeQueue.Finish();

            if (output.RequiresGrad)
            {
                output.Creator = new FlashAttentionOp(q, k, v, output, numHeads, isCausal, this, layerId);
            }

            return output;
        }

        /// <summary>
        /// Ускоренный обратный проход FlashAttention на двух тайловых ядрах (без lCache).
        /// </summary>
        public (Tensor gradQ, Tensor gradK, Tensor gradV) FlashAttentionBackward(
            Tensor gradOutput, Tensor q, Tensor k, Tensor v, Tensor output,
            int numHeads, bool isCausal, string layerId)
        {
            int batchSize = q.Shape[0];
            int seqLen = q.Shape[1];
            int headDim = q.Shape[2] / numHeads;
            float scale = 1.0f / (float)Math.Sqrt(headDim);

            Tensor gradQ = new Tensor(q.Shape, _computeCtx.Context, $"{layerId}_grad_q", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradK = new Tensor(k.Shape, _computeCtx.Context, $"{layerId}_grad_k", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradV = new Tensor(v.Shape, _computeCtx.Context, $"{layerId}_grad_v", MemoryFlags.ReadWrite, requiresGrad: false);
            OpenCLBufferFiller.FillZero(gradQ.DataBuffer, gradQ.Size * sizeof(float));
            OpenCLBufferFiller.FillZero(gradK.DataBuffer, gradK.Size * sizeof(float));
            OpenCLBufferFiller.FillZero(gradV.DataBuffer, gradV.Size * sizeof(float));
            
            // Обязательная сетка под статический тайл TILE_SIZE = 16
            const int TILE_SIZE = 16;
            int numTiles = (seqLen + TILE_SIZE - 1) / TILE_SIZE;
            IntPtr[] localSize = new[] { new IntPtr(TILE_SIZE), new IntPtr(1) };
            IntPtr[] globalSize = new[] { new IntPtr(numTiles * TILE_SIZE), new IntPtr(batchSize * numHeads) };

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            //activeQueue.Finish();
            // -----------------------------------------------------------------
            // ЯДРО 1: Вычисляем gradQ (тайлинг K и V в __local памяти)
            // -----------------------------------------------------------------
            int arg1 = 0;
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(q.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(k.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(v.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(output.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(gradOutput.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(gradQ.DataBuffer);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(seqLen);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(numHeads);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(headDim);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(scale);
            _kernelFlashAttnBwdDq.Arguments[arg1++].SetValue(isCausal ? 1 : 0);

            activeQueue.EnqueueNDRangeKernel(_kernelFlashAttnBwdDq, globalSize, localSize);

            // -----------------------------------------------------------------
            // ЯДРО 2: Вычисляем gradK и gradV (тайлинг Q и dO в __local памяти)
            // -----------------------------------------------------------------
            int arg2 = 0;
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(q.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(k.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(v.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(output.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(gradOutput.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(gradK.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(gradV.DataBuffer);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(seqLen);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(numHeads);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(headDim);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(scale);
            _kernelFlashAttnBwdDkv.Arguments[arg2++].SetValue(isCausal ? 1 : 0);

            activeQueue.EnqueueNDRangeKernel(_kernelFlashAttnBwdDkv, globalSize, localSize);
            //activeQueue.Finish();

            return (gradQ, gradK, gradV);
        }

        #endregion
        #region SwiGLU
        public Tensor SwiGLUGateUp(Tensor x, Tensor weightGateUp, int intermediateDim, string layerId = "swiglu")
        {
            int dModel = x.Shape[x.Shape.Length - 1];
            int totalRows = x.Size / dModel;

            int[] outShape = (int[])x.Shape.Clone();
            outShape[outShape.Length - 1] = intermediateDim;

            bool reqGrad = x.RequiresGrad || weightGateUp.RequiresGrad;
            Tensor output = new Tensor(outShape, _computeCtx.Context, $"{layerId}_out", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            const int TILE = 16;
            IntPtr[] localSize = new[] { new IntPtr(TILE), new IntPtr(TILE) };
            IntPtr[] globalSize = new[]
            {
        new IntPtr(((intermediateDim + TILE - 1) / TILE) * TILE),
        new IntPtr(((totalRows + TILE - 1) / TILE) * TILE)
    };

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelSwiGluForward.Arguments[arg++].SetValue(x.DataBuffer);
            _kernelSwiGluForward.Arguments[arg++].SetValue(weightGateUp.DataBuffer);
            _kernelSwiGluForward.Arguments[arg++].SetValue(output.DataBuffer);
            _kernelSwiGluForward.Arguments[arg++].SetValue(totalRows);
            _kernelSwiGluForward.Arguments[arg++].SetValue(dModel);
            _kernelSwiGluForward.Arguments[arg++].SetValue(intermediateDim);

            activeQueue.EnqueueNDRangeKernel(_kernelSwiGluForward,  globalSize, localSize);
            activeQueue.Finish();

            if (output.RequiresGrad)
            {
                output.Creator = new SwiGLUOp(x, weightGateUp, output, intermediateDim, this, layerId);
            }

            return output;
        }

        public (Tensor gradX, Tensor gradW) SwiGLUGateUpBackward(
            Tensor gradOutput, Tensor x, Tensor weightGateUp, int intermediateDim, string layerId)
        {
            int dModel = x.Shape[x.Shape.Length - 1];
            int totalRows = x.Size / dModel;
            int twoN = 2 * intermediateDim;

            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}_grad_x", MemoryFlags.ReadWrite, requiresGrad: false);
            Tensor gradW = new Tensor(weightGateUp.Shape, _computeCtx.Context, $"{layerId}_grad_w", MemoryFlags.ReadWrite, requiresGrad: false);
            OpenCLBufferFiller.FillZero(gradX.DataBuffer, gradX.Size * sizeof(float));
            OpenCLBufferFiller.FillZero(gradW.DataBuffer, gradW.Size * sizeof(float));
            // Временный GPU-буфер под [d_gate | d_up] формы [totalRows, 2 * intermediateDim]
            long dGateUpBytes = (long)totalRows * twoN * sizeof(float);
            using NOpenCL.Buffer dGateUpBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, dGateUpBytes);

            const int TILE = 16;
            IntPtr[] localSize = new[] { new IntPtr(TILE), new IntPtr(TILE) };
            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            activeQueue.Finish();
            // Шаг 1: Вычисляем dGateUp [M, 2*N]
            IntPtr[] global1 = new[]
            {
        new IntPtr(((intermediateDim + TILE - 1) / TILE) * TILE),
        new IntPtr(((totalRows + TILE - 1) / TILE) * TILE)
    };
            int a1 = 0;
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(gradOutput.DataBuffer);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(x.DataBuffer);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(weightGateUp.DataBuffer);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(dGateUpBuf);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(totalRows);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(dModel);
            _kernelSwiGluComputeDGateDup.Arguments[a1++].SetValue(intermediateDim);
            activeQueue.EnqueueNDRangeKernel(_kernelSwiGluComputeDGateDup, null, global1, localSize);

            // Шаг 2A: Вычисляем gradX [M, K]
            IntPtr[] global2a = new[]
            {
        new IntPtr(((dModel + TILE - 1) / TILE) * TILE),
        new IntPtr(((totalRows + TILE - 1) / TILE) * TILE)
    };
            int a2 = 0;
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(dGateUpBuf);
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(weightGateUp.DataBuffer);
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(gradX.DataBuffer);
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(totalRows);
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(dModel);
            _kernelSwiGluBwdDx.Arguments[a2++].SetValue(twoN);
            activeQueue.EnqueueNDRangeKernel(_kernelSwiGluBwdDx, global2a, localSize);

            // Шаг 2B: Вычисляем gradW [K, 2*N]
            IntPtr[] global2b = new[]
            {
        new IntPtr(((twoN + TILE - 1) / TILE) * TILE),
        new IntPtr(((dModel + TILE - 1) / TILE) * TILE)
    };
            int a3 = 0;
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(x.DataBuffer);
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(dGateUpBuf);
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(gradW.DataBuffer);
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(totalRows);
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(dModel);
            _kernelSwiGluBwdDw.Arguments[a3++].SetValue(twoN);
            activeQueue.EnqueueNDRangeKernel(_kernelSwiGluBwdDw, global2b, localSize);

            activeQueue.Finish();

            return (gradX, gradW);
        }

        #endregion
        public Tensor Add(Tensor a, Tensor b, string opId = "add")
        {
            if (a.Size != b.Size)
                throw new ArgumentException($"Размеры тензоров не совпадают: {a.Size} != {b.Size}");

            bool reqGrad = a.RequiresGrad || b.RequiresGrad;
            Tensor output = new Tensor(a.Shape, _computeCtx.Context, $"{opId}_out", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            int localSize = 256;
            int globalSize = ((a.Size + localSize - 1) / localSize) * localSize;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelTensorAdd.Arguments[arg++].SetValue(a.DataBuffer);
            _kernelTensorAdd.Arguments[arg++].SetValue(b.DataBuffer);
            _kernelTensorAdd.Arguments[arg++].SetValue(output.DataBuffer);
            _kernelTensorAdd.Arguments[arg++].SetValue(a.Size);

            queue.EnqueueNDRangeKernel(_kernelTensorAdd, new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });
            queue.Finish();

            if (output.RequiresGrad)
            {
                output.Creator = new AddOp(a, b, output,this,opId);
            }

            return output;
        }
        public Tensor CloneTensor(Tensor src, string newTensorId)
        {
            Tensor dst = new Tensor(src.Shape, _computeCtx.Context, newTensorId, MemoryFlags.ReadWrite, requiresGrad: false);
            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            queue.EnqueueCopyBuffer(src.DataBuffer, dst.DataBuffer, 0, 0, (long)src.Size * sizeof(float));
            queue.Finish();
            return dst;
        }
        #region MKV
        // Forward: w [C], u [C], k [B, T, C], v [B, T, C] -> y [B, T, C]
        // В NnClGlob.cs:
        public Tensor Wkv(Tensor w, Tensor u, Tensor k, Tensor v, string layerId)
        {
            int B = k.Shape[0];
            int T = k.Shape[1];
            int C = k.Shape[2];

            bool reqGrad = w.RequiresGrad || u.RequiresGrad || k.RequiresGrad || v.RequiresGrad;
            Tensor y = new Tensor(k.Shape, _computeCtx.Context, $"{layerId}.wkv_y", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            long stateBytes = (long)B * T * C * sizeof(float);
            NOpenCL.Buffer stateAa = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"wkv_state_aa_{layerId}", stateBytes);
            NOpenCL.Buffer stateBb = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"wkv_state_bb_{layerId}", stateBytes);
            NOpenCL.Buffer statePp = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"wkv_state_pp_{layerId}", stateBytes);

            int totalThreads = B * C;
            int localSize = 64;
            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelWkvForward.Arguments[arg++].SetValue(w.DataBuffer);
            _kernelWkvForward.Arguments[arg++].SetValue(u.DataBuffer);
            _kernelWkvForward.Arguments[arg++].SetValue(k.DataBuffer);
            _kernelWkvForward.Arguments[arg++].SetValue(v.DataBuffer);
            _kernelWkvForward.Arguments[arg++].SetValue(y.DataBuffer);
            _kernelWkvForward.Arguments[arg++].SetValue(stateAa);
            _kernelWkvForward.Arguments[arg++].SetValue(stateBb);
            _kernelWkvForward.Arguments[arg++].SetValue(statePp);
            _kernelWkvForward.Arguments[arg++].SetValue(B);
            _kernelWkvForward.Arguments[arg++].SetValue(T);
            _kernelWkvForward.Arguments[arg++].SetValue(C);

            queue.EnqueueNDRangeKernel(_kernelWkvForward, null, new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });

            if (reqGrad)
            {
                y.Creator = new WkvOp(w, u, k, v, y, stateAa, stateBb, statePp, this, layerId);
            }

            return y;
        }

        // Backward
        public (Tensor gw, Tensor gu, Tensor gk, Tensor gv) WkvBackward(
            Tensor w, Tensor u, Tensor k, Tensor v, Tensor y, Tensor gradOutput,
            NOpenCL.Buffer stateAa, NOpenCL.Buffer stateBb, NOpenCL.Buffer statePp, string layerId)
        {
            int B = k.Shape[0];
            int T = k.Shape[1];
            int C = k.Shape[2];

            Tensor gw = new Tensor(w.Shape, _computeCtx.Context, $"{layerId}.gw", MemoryFlags.ReadWrite, false);
            Tensor gu = new Tensor(u.Shape, _computeCtx.Context, $"{layerId}.gu", MemoryFlags.ReadWrite, false);
            Tensor gk = new Tensor(k.Shape, _computeCtx.Context, $"{layerId}.gk", MemoryFlags.ReadWrite, false);
            Tensor gv = new Tensor(v.Shape, _computeCtx.Context, $"{layerId}.gv", MemoryFlags.ReadWrite, false);

            long partBytes = (long)B * C * sizeof(float);
            using NOpenCL.Buffer gwPart = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, partBytes);
            using NOpenCL.Buffer guPart = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, partBytes);

            int totalThreads = B * C;
            int localSize = 64;
            int globalSize = ((totalThreads + localSize - 1) / localSize) * localSize;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelWkvBackward.Arguments[arg++].SetValue(w.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(u.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(k.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(v.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(y.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(gradOutput.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(stateAa);
            _kernelWkvBackward.Arguments[arg++].SetValue(stateBb);
            _kernelWkvBackward.Arguments[arg++].SetValue(statePp);
            _kernelWkvBackward.Arguments[arg++].SetValue(gwPart);
            _kernelWkvBackward.Arguments[arg++].SetValue(guPart);
            _kernelWkvBackward.Arguments[arg++].SetValue(gk.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(gv.DataBuffer);
            _kernelWkvBackward.Arguments[arg++].SetValue(B);
            _kernelWkvBackward.Arguments[arg++].SetValue(T);
            _kernelWkvBackward.Arguments[arg++].SetValue(C);

            queue.EnqueueNDRangeKernel(_kernelWkvBackward, null, new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });

            // Редукция по батчу B с помощью вашего готового ядра _kernelReduceRowsSum
            int redLocal = 64;
            int redGlobal = ((C + redLocal - 1) / redLocal) * redLocal;

            // Суммируем gwPart [B, C] -> gw [C]
            _kernelReduceRowsSum.Arguments[0].SetValue(gwPart);
            _kernelReduceRowsSum.Arguments[1].SetValue(gw.DataBuffer);
            _kernelReduceRowsSum.Arguments[2].SetValue(B);
            _kernelReduceRowsSum.Arguments[3].SetValue(C);
            queue.EnqueueNDRangeKernel(_kernelReduceRowsSum, null, new[] { new IntPtr(redGlobal) }, new[] { new IntPtr(redLocal) });

            // Суммируем guPart [B, C] -> gu [C]
            _kernelReduceRowsSum.Arguments[0].SetValue(guPart);
            _kernelReduceRowsSum.Arguments[1].SetValue(gu.DataBuffer);
            _kernelReduceRowsSum.Arguments[2].SetValue(B);
            _kernelReduceRowsSum.Arguments[3].SetValue(C);
            queue.EnqueueNDRangeKernel(_kernelReduceRowsSum, null, new[] { new IntPtr(redGlobal) }, new[] { new IntPtr(redLocal) });

            return (gw, gu, gk, gv);
        }
        public Tensor TokenShift(Tensor x, Tensor mu, string layerId)
        {
            int B = x.Shape[0], T = x.Shape[1], C = x.Shape[2];
            bool reqGrad = x.RequiresGrad || mu.RequiresGrad;
            Tensor outT = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}.tshift", MemoryFlags.ReadWrite, reqGrad);

            int total = B * T * C;
            int local = 256;
            int global = ((total + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelTokenShiftFwd.Arguments[0].SetValue(x.DataBuffer);
            _kernelTokenShiftFwd.Arguments[1].SetValue(mu.DataBuffer);
            _kernelTokenShiftFwd.Arguments[2].SetValue(outT.DataBuffer);
            _kernelTokenShiftFwd.Arguments[3].SetValue(B);
            _kernelTokenShiftFwd.Arguments[4].SetValue(T);
            _kernelTokenShiftFwd.Arguments[5].SetValue(C);
            q.EnqueueNDRangeKernel(_kernelTokenShiftFwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            if (reqGrad)
                outT.Creator = new TokenShiftOp(x, mu, outT, this, layerId);

            return outT;
        }

        public (Tensor gradX, Tensor gradMu) TokenShiftBackward(Tensor gradOut, Tensor x, Tensor mu, string layerId)
        {
            int B = x.Shape[0], T = x.Shape[1], C = x.Shape[2];
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}.tshift_gx", MemoryFlags.ReadWrite, false);
            Tensor gradMu = new Tensor(mu.Shape, _computeCtx.Context, $"{layerId}.tshift_gmu", MemoryFlags.ReadWrite, false);

            long partBytes = (long)B * T * C * sizeof(float);
            NOpenCL.Buffer gradMuPart = OpenCLBufferManager.GetOrCreateBuffer(_computeCtx.Context, $"{layerId}.tshift_gmupart", partBytes);

            int total = B * T * C;
            int local = 256;
            int global = ((total + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelTokenShiftBwd.Arguments[0].SetValue(x.DataBuffer);
            _kernelTokenShiftBwd.Arguments[1].SetValue(mu.DataBuffer);
            _kernelTokenShiftBwd.Arguments[2].SetValue(gradOut.DataBuffer);
            _kernelTokenShiftBwd.Arguments[3].SetValue(gradX.DataBuffer);
            _kernelTokenShiftBwd.Arguments[4].SetValue(gradMuPart);
            _kernelTokenShiftBwd.Arguments[5].SetValue(B);
            _kernelTokenShiftBwd.Arguments[6].SetValue(T);
            _kernelTokenShiftBwd.Arguments[7].SetValue(C);
            q.EnqueueNDRangeKernel(_kernelTokenShiftBwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            // Редукция [B * T, C] -> [C] через существующее _kernelReduceRowsSum
            int redLocal = 64;
            int redGlobal = ((C + redLocal - 1) / redLocal) * redLocal;
            _kernelReduceRowsSum.Arguments[0].SetValue(gradMuPart);
            _kernelReduceRowsSum.Arguments[1].SetValue(gradMu.DataBuffer);
            _kernelReduceRowsSum.Arguments[2].SetValue(B * T);
            _kernelReduceRowsSum.Arguments[3].SetValue(C);
            q.EnqueueNDRangeKernel(_kernelReduceRowsSum, null, new[] { (IntPtr)redGlobal }, new[] { (IntPtr)redLocal });

            return (gradX, gradMu);
        }

        // 2. Унарные операции RWKV (0 = Sigmoid, 1 = SquaredReLU, 2 = NegExp)
        public Tensor RwkvUnary(Tensor x, int mode, string layerId)
        {
            Tensor y = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}.unary_{mode}", MemoryFlags.ReadWrite, x.RequiresGrad);
            int local = 256;
            int global = ((x.Size + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelRwkvUnaryFwd.Arguments[0].SetValue(x.DataBuffer);
            _kernelRwkvUnaryFwd.Arguments[1].SetValue(y.DataBuffer);
            _kernelRwkvUnaryFwd.Arguments[2].SetValue(x.Size);
            _kernelRwkvUnaryFwd.Arguments[3].SetValue(mode);
            q.EnqueueNDRangeKernel(_kernelRwkvUnaryFwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            if (x.RequiresGrad)
                y.Creator = new RwkvUnaryOp(x, y, mode, this, layerId);

            return y;
        }

        public Tensor RwkvUnaryBackward(Tensor gradY, Tensor x, Tensor y, int mode, string layerId)
        {
            Tensor gradX = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}.unary_gx_{mode}", MemoryFlags.ReadWrite, false);
            int local = 256;
            int global = ((x.Size + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelRwkvUnaryBwd.Arguments[0].SetValue(x.DataBuffer);
            _kernelRwkvUnaryBwd.Arguments[1].SetValue(y.DataBuffer);
            _kernelRwkvUnaryBwd.Arguments[2].SetValue(gradY.DataBuffer);
            _kernelRwkvUnaryBwd.Arguments[3].SetValue(gradX.DataBuffer);
            _kernelRwkvUnaryBwd.Arguments[4].SetValue(x.Size);
            _kernelRwkvUnaryBwd.Arguments[5].SetValue(mode);
            q.EnqueueNDRangeKernel(_kernelRwkvUnaryBwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            return gradX;
        }

        // 3. Поэлементное умножение (a * b)
        public Tensor Mul(Tensor a, Tensor b, string layerId)
        {
            bool reqGrad = a.RequiresGrad || b.RequiresGrad;
            Tensor c = new Tensor(a.Shape, _computeCtx.Context, $"{layerId}.mul", MemoryFlags.ReadWrite, reqGrad);
            int local = 256;
            int global = ((a.Size + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelTensorMulFwd.Arguments[0].SetValue(a.DataBuffer);
            _kernelTensorMulFwd.Arguments[1].SetValue(b.DataBuffer);
            _kernelTensorMulFwd.Arguments[2].SetValue(c.DataBuffer);
            _kernelTensorMulFwd.Arguments[3].SetValue(a.Size);
            q.EnqueueNDRangeKernel(_kernelTensorMulFwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            if (reqGrad)
                c.Creator = new MulOp(a, b, c, this, layerId);

            return c;
        }

        public (Tensor gradA, Tensor gradB) MulBackward(Tensor gradC, Tensor a, Tensor b, string layerId)
        {
            Tensor gradA = new Tensor(a.Shape, _computeCtx.Context, $"{layerId}.mul_ga", MemoryFlags.ReadWrite, false);
            Tensor gradB = new Tensor(b.Shape, _computeCtx.Context, $"{layerId}.mul_gb", MemoryFlags.ReadWrite, false);
            int local = 256;
            int global = ((a.Size + local - 1) / local) * local;

            var q = DeviceQueueRegistry.GetCurrent();
            _kernelTensorMulBwd.Arguments[0].SetValue(a.DataBuffer);
            _kernelTensorMulBwd.Arguments[1].SetValue(b.DataBuffer);
            _kernelTensorMulBwd.Arguments[2].SetValue(gradC.DataBuffer);
            _kernelTensorMulBwd.Arguments[3].SetValue(gradA.DataBuffer);
            _kernelTensorMulBwd.Arguments[4].SetValue(gradB.DataBuffer);
            _kernelTensorMulBwd.Arguments[5].SetValue(a.Size);
            q.EnqueueNDRangeKernel(_kernelTensorMulBwd, null, new[] { (IntPtr)global }, new[] { (IntPtr)local });

            return (gradA, gradB);
        }
        #endregion
        private static readonly Random _dropoutRng = new Random();

        /// <summary>
        /// Inverted Dropout (Forward).
        /// В режиме инференса (isTraining = false) возвращает исходный тензор без затрат VRAM и GPU.
        /// </summary>
        public Tensor Dropout(Tensor x, float p = 0.1f, bool isTraining = true, string layerId = "dropout")
        {
            if (p < 0.0f || p >= 1.0f)
                throw new ArgumentOutOfRangeException(nameof(p), "Вероятность Dropout должна быть в диапазоне [0.0, 1.0).");

            // При инференсе или p == 0 слой прозрачен
            if (!isTraining || p == 0.0f)
                return x;

            float scale = 1.0f / (1.0f - p);
            uint seed = (uint)_dropoutRng.Next();
            int size = (int)x.Size;

            Tensor y = new Tensor(x.Shape, _computeCtx.Context, $"{layerId}_out", MemoryFlags.ReadWrite, requiresGrad: x.RequiresGrad);

            // Выделяем компактный буфер маски (1 байт на элемент)
            NOpenCL.Buffer maskBuf = _computeCtx.Context.CreateBuffer(MemoryFlags.ReadWrite, size * sizeof(byte));

            int localSize = 256;
            int globalSize = ((size + localSize - 1) / localSize) * localSize;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelDropoutForward.Arguments[arg++].SetValue(x.DataBuffer);
            _kernelDropoutForward.Arguments[arg++].SetValue(y.DataBuffer);
            _kernelDropoutForward.Arguments[arg++].SetValue(maskBuf);
            _kernelDropoutForward.Arguments[arg++].SetValue(p);
            _kernelDropoutForward.Arguments[arg++].SetValue(scale);
            _kernelDropoutForward.Arguments[arg++].SetValue(seed);
            _kernelDropoutForward.Arguments[arg++].SetValue(size);

            activeQueue.EnqueueNDRangeKernel(_kernelDropoutForward,  new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });
            activeQueue.Finish();

            if (y.RequiresGrad)
            {
                var buffers = new LayerBuffers();
                buffers.layerBuffers.Add($"{layerId}_mask", maskBuf);
                GradientStorage.Push(buffers);

                y.Creator = new DropoutOp(x, y, scale, this, layerId);
            }
            else
            {
                maskBuf.Dispose();
            }

            return y;
        }

        /// <summary>
        /// Inverted Dropout (Backward).
        /// </summary>
        public Tensor DropoutBackward(Tensor gradOutput, NOpenCL.Buffer maskBuf, float scale, string layerId)
        {
            int size = (int)gradOutput.Size;
            Tensor gradX = new Tensor(gradOutput.Shape, _computeCtx.Context, $"{layerId}_grad_x", MemoryFlags.ReadWrite, requiresGrad: false);

            int localSize = 256;
            int globalSize = ((size + localSize - 1) / localSize) * localSize;

            CommandQueue activeQueue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelDropoutBackward.Arguments[arg++].SetValue(gradOutput.DataBuffer);
            _kernelDropoutBackward.Arguments[arg++].SetValue(maskBuf);
            _kernelDropoutBackward.Arguments[arg++].SetValue(gradX.DataBuffer);
            _kernelDropoutBackward.Arguments[arg++].SetValue(scale);
            _kernelDropoutBackward.Arguments[arg++].SetValue(size);

            activeQueue.EnqueueNDRangeKernel(_kernelDropoutBackward,  new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });
            activeQueue.Finish();

            return gradX;
        }
        /// <summary>
        /// Нарезает изображение [B, C, H, W] на патчи [B, (H/P)*(W/P), C*P*P].
        /// </summary>
        public Tensor Patchify(Tensor img, int patchSize, string opId = "patchify")
        {
            if (img.Shape.Length != 4)
                throw new ArgumentException("Ожидается 4D-тензор изображения [Batch, Channels, Height, Width].");

            int B = img.Shape[0];
            int C = img.Shape[1];
            int H = img.Shape[2];
            int W = img.Shape[3];

            if (H % patchSize != 0 || W % patchSize != 0)
                throw new ArgumentException($"Размеры изображения ({H}x{W}) должны делиться на patchSize ({patchSize}).");

            int numPatches = (H / patchSize) * (W / patchSize);
            int patchDim = C * patchSize * patchSize;

            Tensor output = new Tensor(new[] { B, numPatches, patchDim }, _computeCtx.Context, $"{opId}_out", MemoryFlags.ReadWrite, requiresGrad: img.RequiresGrad);

            const int TILE = 16;
            IntPtr[] localSize = new[] { new IntPtr(TILE), new IntPtr(TILE) };
            IntPtr[] globalSize = new[]
            {
                new IntPtr(((patchDim + TILE - 1) / TILE) * TILE),
                new IntPtr(((B * numPatches + TILE - 1) / TILE) * TILE)
            };

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelPatchifyForward.Arguments[arg++].SetValue(img.DataBuffer);
            _kernelPatchifyForward.Arguments[arg++].SetValue(output.DataBuffer);
            _kernelPatchifyForward.Arguments[arg++].SetValue(B);
            _kernelPatchifyForward.Arguments[arg++].SetValue(C);
            _kernelPatchifyForward.Arguments[arg++].SetValue(H);
            _kernelPatchifyForward.Arguments[arg++].SetValue(W);
            _kernelPatchifyForward.Arguments[arg++].SetValue(patchSize);

            queue.EnqueueNDRangeKernel(_kernelPatchifyForward, null, globalSize, localSize);
            queue.Finish();

            if (output.RequiresGrad)
            {
                output.Creator = new PatchifyOp(img, output, patchSize, this, opId);
            }

            return output;
        }

        public Tensor PatchifyBackward(Tensor gradOutput, int[] imgShape, int patchSize, string opId)
        {
            int B = imgShape[0];
            int C = imgShape[1];
            int H = imgShape[2];
            int W = imgShape[3];
            int numPatches = (H / patchSize) * (W / patchSize);
            int patchDim = C * patchSize * patchSize;

            Tensor gradImg = new Tensor(imgShape, _computeCtx.Context, $"{opId}_grad_img", MemoryFlags.ReadWrite, requiresGrad: false);

            const int TILE = 16;
            IntPtr[] localSize = new[] { new IntPtr(TILE), new IntPtr(TILE) };
            IntPtr[] globalSize = new[]
            {
                new IntPtr(((patchDim + TILE - 1) / TILE) * TILE),
                new IntPtr(((B * numPatches + TILE - 1) / TILE) * TILE)
            };

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelPatchifyBackward.Arguments[arg++].SetValue(gradOutput.DataBuffer);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(gradImg.DataBuffer);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(B);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(C);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(H);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(W);
            _kernelPatchifyBackward.Arguments[arg++].SetValue(patchSize);

            queue.EnqueueNDRangeKernel(_kernelPatchifyBackward, null, globalSize, localSize);
            queue.Finish();

            return gradImg;
        }

        /// <summary>
        /// Прибавляет позиционные эмбеддинги [1, NumPatches, D] к тензору [B, NumPatches, D].
        /// </summary>
        public Tensor AddPosEmbed(Tensor x, Tensor posEmbed, string opId = "pos_emb")
        {
            int B = x.Shape[0];
            int nd = (int)(x.Size / B);
            if (posEmbed.Size != nd)
                throw new ArgumentException($"Размер PosEmbed ({posEmbed.Size}) должен совпадать с NumPatches * D ({nd}).");

            bool reqGrad = x.RequiresGrad || posEmbed.RequiresGrad;
            Tensor output = new Tensor(x.Shape, _computeCtx.Context, $"{opId}_out", MemoryFlags.ReadWrite, requiresGrad: reqGrad);

            int totalSize = (int)x.Size;
            int localSize = 256;
            int globalSize = ((totalSize + localSize - 1) / localSize) * localSize;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelAddPosForward.Arguments[arg++].SetValue(x.DataBuffer);
            _kernelAddPosForward.Arguments[arg++].SetValue(posEmbed.DataBuffer);
            _kernelAddPosForward.Arguments[arg++].SetValue(output.DataBuffer);
            _kernelAddPosForward.Arguments[arg++].SetValue(B);
            _kernelAddPosForward.Arguments[arg++].SetValue(nd);

            queue.EnqueueNDRangeKernel(_kernelAddPosForward, new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });
            queue.Finish();

            if (output.RequiresGrad)
            {
                output.Creator = new AddPosEmbedOp(x, posEmbed, output, this, opId);
            }

            return output;
        }

        public (Tensor gradX, Tensor gradPos) AddPosEmbedBackward(Tensor gradOutput, Tensor posEmbed, string opId)
        {
            int B = gradOutput.Shape[0];
            int nd = (int)(gradOutput.Size / B);

            // Градиент по x — это полная копия gradOutput
            Tensor gradX = CloneTensor(gradOutput, $"{opId}_grad_x");
            // Градиент по posEmbed — сумма gradOutput по всем элементам батча B
            Tensor gradPos = new Tensor(posEmbed.Shape, _computeCtx.Context, $"{opId}_grad_pos", MemoryFlags.ReadWrite, requiresGrad: false);

            int localSize = 256;
            int globalSize = ((nd + localSize - 1) / localSize) * localSize;

            CommandQueue queue = DeviceQueueRegistry.GetCurrent();
            int arg = 0;
            _kernelAddPosBackwardPos.Arguments[arg++].SetValue(gradOutput.DataBuffer);
            _kernelAddPosBackwardPos.Arguments[arg++].SetValue(gradPos.DataBuffer);
            _kernelAddPosBackwardPos.Arguments[arg++].SetValue(B);
            _kernelAddPosBackwardPos.Arguments[arg++].SetValue(nd);

            queue.EnqueueNDRangeKernel(_kernelAddPosBackwardPos,  new[] { new IntPtr(globalSize) }, new[] { new IntPtr(localSize) });
            queue.Finish();

            return (gradX, gradPos);
        }
    }

}


