import numpy as np
import pyopencl as cl

class Tensor:
    def __init__(self, shape, ctx, queue, data_buffer=None, grad_buffer=None, dtype=np.float32):
        """
        Базовый многомерный тензор, хранящийся как плоский 1D OpenCL-буфер.
        
        :param shape: tuple или list, задающий размерности (например, [BATCH, SEQ_LEN, D_MODEL])
        :param ctx: pyopencl.Context
        :param queue: pyopencl.CommandQueue
        :param data_buffer: Существующий cl.Buffer (для виртуальных операций вроде transpose/reshape)
        :param grad_buffer: Буфер для градиентов (нужен в режиме обучения)
        """
        self.ctx = ctx
        self.queue = queue
        self.dtype = np.dtype(dtype)
        
        # Динамические метаданные на стороне Python (меняются без перевыделения VRAM)
        self.shape = list(shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))  # Общее число элементов
        self.nbytes = self.size * self.dtype.itemsize

        # Управление физической памятью GPU (1D плоские буферы)
        if data_buffer is not None:
            self.data_buffer = data_buffer
        else:
            # Выделяем пустой буфер во VRAM под данные
            self.data_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.nbytes)
            # Инициализируем нулями (опционально)
            cl.enqueue_fill_buffer(self.queue, self.data_buffer, np.float32(0.0), 0, self.nbytes)

        # Буфер градиентов (выделяется лениво при включении autograd/training)
        self.grad_buffer = grad_buffer

    @classmethod
    def from_numpy(cls, np_array, ctx, queue):
        """Фабричный метод создания тензора из массива NumPy с копированием во VRAM"""
        np_array = np_array.astype(np.float32)
        nbytes = np_array.nbytes
        data_buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np_array)
        return cls(shape=np_array.shape, ctx=ctx, queue=queue, data_buffer=data_buffer)

    def to_numpy(self):
        """HostReturner: Перевод (скачивание) данных из VRAM обратно в RAM CPU"""
        host_array = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.queue, host_array, self.data_buffer)
        self.queue.finish()  # Встроенная синхронизация, гарантирующая актуальность данных
        return host_array

    def reshape(self, *new_shape):
        """
        ВИРТУАЛЬНАЯ ОПЕРАЦИЯ: Меняет только метаданные осей. 
        Новая память во VRAM НЕ выделяется, указатель на буфер остается прежним.
        """
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]
            
        # Обработка -1 в размерностях (как в PyTorch)
        new_shape = list(new_shape)
        if -1 in new_shape:
            idx = new_shape.index(-1)
            current_prod = int(np.prod([d for d in new_shape if d != -1]))
            new_shape[idx] = self.size // current_prod

        assert int(np.prod(new_shape)) == self.size, f"Невозможно сделать reshape из {self.shape} в {new_shape}"
        
        # Возвращаем новый объект Тензора, разделяющий тот же физический буфер данных
        return Tensor(shape=new_shape, ctx=self.ctx, queue=self.queue, 
                      data_buffer=self.data_buffer, grad_buffer=self.grad_buffer)

    def unsqueeze(self, dim):
        """ВИРТУАЛЬНАЯ ОПЕРАЦИЯ: Вставляет единичную ось в указанную позицию формы"""
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return self.reshape(new_shape)

    def squeeze(self, dim=None):
        """ВИРТУАЛЬНАЯ ОПЕРАЦИЯ: Удаляет единичные оси"""
        if dim is not None:
            if self.shape[dim] == 1:
                new_shape = list(self.shape)
                new_shape.pop(dim)
                return self.reshape(new_shape)
            return self
        else:
            new_shape = [d for d in self.shape if d != 1]
            return self.reshape(new_shape)

    def require_grad(self):
        """Включает отслеживание градиентов для обучения (выделяет буфер под backward)"""
        if self.grad_buffer is None:
            self.grad_buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.nbytes)
            cl.enqueue_fill_buffer(self.queue, self.grad_buffer, np.float32(0.0), 0, self.nbytes)

    def __repr__(self):
        return f"nncl.Tensor(shape={self.shape}, dtype={self.dtype.name}, device=OpenCL)"
