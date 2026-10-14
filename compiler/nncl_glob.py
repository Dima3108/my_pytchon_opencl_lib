import os
import numpy as np
import pyopencl as cl
from core.tensor import Tensor
from compiler.nncl_compiler import JITProgramCache

class NnClGlob:
    def __init__(self, ctx, queue):
        """
        Диспетчер тяжелых глобальных операций библиотеки nncl.
        Загружает и кэширует скомпилированные Си-ядра.
        """
        self.ctx = ctx
        self.queue = queue
        self._load_kernels()

    def _load_kernels(self):
        """Загружает Си-код всех тяжелых ядер из cl_kernels"""
        kernels_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cl_kernels'))
        
        # 1. Загрузка RMSNorm
        with open(os.path.join(kernels_dir, 'rmsnorm.cl'), 'r', encoding='utf-8') as f:
            self.prg_rmsnorm = JITProgramCache.get_or_compile(self.ctx, f.read())
            
        # 2. Загрузка Softmax
        with open(os.path.join(kernels_dir, 'Softmax.cl'), 'r', encoding='utf-8') as f:
            self.prg_softmax = JITProgramCache.get_or_compile(self.ctx, f.read())
            
        # 3. Загрузка Matmul / BMM
        with open(os.path.join(kernels_dir, 'matmul.cl'), 'r', encoding='utf-8') as f:
            self.prg_matmul = JITProgramCache.get_or_compile(self.ctx, f.read())
            
        # 4. Загрузка Permute (Transpose)
        with open(os.path.join(kernels_dir, 'permute.cl'), 'r', encoding='utf-8') as f:
            self.prg_permute = JITProgramCache.get_or_compile(self.ctx, f.read())

        # 5. Загрузка CausalConv1d
        with open(os.path.join(kernels_dir, 'CausalConv1d.cl'), 'r', encoding='utf-8') as f:
            self.prg_conv = JITProgramCache.get_or_compile(self.ctx, f.read())

    def rmsnorm(self, x: Tensor, gamma: Tensor) -> Tensor:
        """
        Глобальная операция RMSNorm (Forward)
        Обрабатывает тензор любой формы, считая нормализацию по последней оси d_model
        """
        d_model = x.shape[-1]
        # Вычисляем количество строк (B * L)
        num_rows = x.size // d_model
        
        # Создаем выходной тензор и кэш инвертированного RMS для будущего backward
        y = Tensor(x.shape, self.ctx, self.queue)
        inv_rms_cache = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, num_rows * 4) # float32 = 4 bytes
        
        # Настройка сетки потоков
        # Каждая рабочая группа из 256 потоков обрабатывает одну строку
        local_size = (256,)
        global_size = (num_rows * 256,)
        
        self.prg_rmsnorm.rmsnorm_forward(
            self.queue, global_size, local_size,
            x.data_buffer, gamma.data_buffer, y.data_buffer, inv_rms_cache,
            np.int32(d_model)
        )
        self.queue.finish()
        
        # Сохраняем кэш редукции прямо внутри тензора (нужно для обучения)
        y._inv_rms_cache = inv_rms_cache
        return y

    def softmax(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Глобальная операция Safe Softmax (Forward)
        Рассчитана на работу со скорами внимания [B * H * L, SEQ_LEN]
        """
        assert dim == -1 or dim == len(x.shape) - 1, "Текущая реализация ядра Softmax поддерживает редукцию только по последней оси."
        
        seq_len = x.shape[-1]
        len_=seq_len
        print(seq_len)
        num_rows = x.size // seq_len
        
        # Создаем выходной тензор (память во VRAM выделится автоматически)
        y = Tensor(x.shape, self.ctx, self.queue)
        
        # Настройка сетки потоков
        local_size = (256,)
        global_size = (num_rows * 256,)
        
        # Запускаем ядро, явно проверяя, что передаем нативные cl.Buffer объекты
        self.prg_softmax.softmax_forward(
            self.queue, 
            global_size, 
            local_size,
            x.data_buffer, 
            y.data_buffer, 
            np.int32(len_)
        )
        self.queue.finish()
        return y


    def bmm(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Пакетное матричное умножение (Batch Matrix Multiplication)
        A: [Batch, M, K], B: [Batch, K, N] -> Выход: [Batch, M, N]
        """
        assert len(A.shape) == 3 and len(B.shape) == 3, "BMM ожидает строго 3D-тензоры [Batch, M/K, K/N]"
        assert A.shape[0] == B.shape[0], "Размерность Batch у матриц A и B должна совпадать"
        assert A.shape[2] == B.shape[1], f"Несовпадение внутренних осей матриц: K={A.shape[2]} и K={B.shape[1]}"
        
        Batch = A.shape[0]
        M, K = A.shape[1], A.shape[2]
        N = B.shape[3] if len(B.shape) > 3 else B.shape[2]
        
        C = Tensor([Batch, M, N], self.ctx, self.queue)
        
        # Шаги смещения внутри плоского буфера для каждой матрицы пакета
        batch_stride_A = M * K
        batch_stride_B = K * N
        batch_stride_C = M * N
        
        # Настройка 3D-сетки потоков (под тайлы 16х16)
        local_size = (16, 16, 1)
        global_size_x = int(np.ceil(N / 16.0) * 16)
        global_size_y = int(np.ceil(M / 16.0) * 16)
        global_size_z = Batch
        global_size = (global_size_x, global_size_y, global_size_z)
        
        self.prg_matmul.bmm_tiled(
            self.queue, global_size, local_size,
            A.data_buffer, B.data_buffer, C.data_buffer,
            np.int32(M), np.int32(K), np.int32(N),
            np.int32(batch_stride_A), np.int32(batch_stride_B), np.int32(batch_stride_C)
        )
        self.queue.finish()
        return C

    def permute(self, x: Tensor, axes: list) -> Tensor:
        """
        Глобальная перестановка осей тензора (Универсальный Permute/Transpose до 10D)
        """
        assert len(axes) == len(x.shape), "Количество осей перестановки должно совпадать с ndim тензора"
        
        # Вычисляем новую форму выходного тензора
        dst_shape = [x.shape[i] for i in axes]
        y = Tensor(dst_shape, self.ctx, self.queue)
        
        # Переводим метаданные в C-массивы int32
        src_dims_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(x.shape, dtype=np.int32))
        dst_dims_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(dst_shape, dtype=np.int32))
        axes_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(axes, dtype=np.int32))
        
        global_size = (x.size,)
        
        self.prg_permute.tensor_permute(
            self.queue, global_size, None,
            x.data_buffer, y.data_buffer,
            src_dims_g, dst_dims_g, axes_g,
            np.int32(x.ndim), np.int32(x.size)
        )
        self.queue.finish()
        return y

    def causal_conv1d(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """
        Каузальная Depthwise свертка (Forward)
        x: [Batch, Channels, Length]
        weight: [Channels, Kernel_Size]
        bias: [Channels]
        """
        Batch, Channels, Length = x.shape
        kernel_size = weight.shape[1]
        
        y = Tensor(x.shape, self.ctx, self.queue)
        
        # 3D сетка потоков: по всей временной оси, по каналам, по батчу
        global_size = (Length, Channels, Batch)
        
        self.prg_conv.causal_conv1d_depthwise(
            self.queue, global_size, None,
            x.data_buffer, weight.data_buffer, bias.data_buffer, y.data_buffer,
            np.int32(Batch), np.int32(Channels), np.int32(Length), np.int32(kernel_size)
        )
        self.queue.finish()
        return y
    def rmsnorm_backward(self, grad_output: Tensor, x: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
        """
        Обратный проход глобальной операции RMSNorm.
        
        :param grad_output: Пришедший сверху градиент dL/dY
        :param x: Исходный входной тензор X из прямого прохода
        :param gamma: Обучаемые веса слоя gamma
        :return: Кортеж (grad_input, grad_gamma)
        """
        d_model = x.shape[-1]
        num_rows = x.size // d_model
        
        # 1. Выделяем тензоры для градиентов по входу (X) и весам (gamma)
        grad_input = Tensor(x.shape, self.ctx, self.queue)
        grad_gamma = Tensor(gamma.shape, self.ctx, self.queue)
        
        # Вспомогательный буфер для накопления градиентов по gamma от каждой строки
        grad_gamma_part = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, x.size * 4) # float32
        
        # Извлекаем сохраненный на этапе forward кэш inv_rms
        # Если вызов идет из цепочки модели, кэш привязан к промежуточному выходу прямого прохода
        assert hasattr(grad_output, '_inv_rms_cache') or hasattr(x, '_inv_rms_cache'), \
            "Кэш inv_rms_cache не найден. Прямой проход rmsnorm должен быть выполнен до backward."
        inv_rms_cache = getattr(grad_output, '_inv_rms_cache', getattr(x, '_inv_rms_cache', None))

        # 2. Запуск Си-ядра вычисления градиента по входу
        local_size = (256,)
        global_size = (num_rows * 256,)
        
        self.prg_rmsnorm.rmsnorm_backward(
            self.queue, global_size, local_size,
            grad_output.data_buffer, x.data_buffer, gamma.data_buffer, inv_rms_cache,
            grad_input.data_buffer, grad_gamma_part,
            np.int32(d_model)
        )
        self.queue.finish()
        
        # 3. Редукция градиента по обучаемым весам gamma
        # Так как веса gamma общие для всего батча и последовательности, нам нужно 
        # просуммировать промежуточный буфер grad_gamma_part по оси строк.
        # Для простоты и точности сделаем редукцию на GPU или вызовем быстрое ядро.
        # В рамках мета-программы на этапе тестов мы можем временно саккумулировать это:
        # (В продакшене LLM здесь вызывается простое редукционное ядро суммы столбцов матрицы)
        host_parts = np.empty((num_rows, d_model), dtype=np.float32)
        cl.enqueue_copy(self.queue, host_parts, grad_gamma_part)
        self.queue.finish()
        
        # Суммируем градиенты от всех токенов в единый вектор градиента весов
        final_grad_gamma = np.sum(host_parts, axis=0, dtype=np.float32)
        
        # Загружаем финальный градиент обратно во VRAM буфер grad_gamma
        cl.enqueue_copy(self.queue, grad_gamma.data_buffer, final_grad_gamma)
        self.queue.finish()
        
        return grad_input, grad_gamma
    def softmax_backward(self, grad_output: Tensor, y: Tensor) -> Tensor:
        """
        Обратный проход глобальной операции Softmax.
        Использует уже вычисленный forward-выход тензора Y для экономии памяти VRAM.
        
        :param grad_output: Пришедший сверху градиент dL/dY [B * H * L, SEQ_LEN]
        :param y: Выход прямого прохода Softmax (вероятности) [B * H * L, SEQ_LEN]
        :return: Выходной градиент dL/dX [B * H * L, SEQ_LEN]
        """
        seq_len = y.shape[-1]
        num_rows = y.size // seq_len
        
        # Выделяем тензор для результирующего градиента по входу
        grad_input = Tensor(y.shape, self.ctx, self.queue)
        
        # Настройка сетки потоков (идентично forward шагу)
        local_size = (256,)
        global_size = (num_rows * 256,)
        
        self.prg_softmax.softmax_backward(
            self.queue, global_size, local_size,
            grad_output.data_buffer, y.data_buffer, grad_input.data_buffer,
            np.int32(seq_len)
        )
        self.queue.finish()
        
        return grad_input
