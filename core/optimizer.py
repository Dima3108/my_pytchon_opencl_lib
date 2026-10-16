import os
import numpy as np
import pyopencl as cl
from compiler.nncl_compiler import JITProgramCache

class SGDOptimizer:
    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        """
        Оптимизатор SGD для обновления весов прямо на GPU.
        
        :param parameters: Список или итератор объектов Tensor (веса модели)
        :param lr: Скорость обучения (Learning Rate)
        :param weight_decay: Коэффициент L2 регуляризации
        """
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        
        # Берем контекст и очередь у первого попавшегося тензора
        if not self.parameters:
            raise ValueError("Список параметров для оптимизатора не должен быть пустым.")
            
        self.ctx = self.parameters[0].ctx
        self.queue = self.parameters[0].queue
        
        # Определяем размер варпа динамически (как в LocalRunner)
        dummy_code = "__kernel void dummy() {}"
        dummy_prg = cl.Program(self.ctx, dummy_code).build()
        try:
            self.warp_size = dummy_prg.dummy.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                self.queue.device
            )
        except Exception:
            self.warp_size = 32

        # Загружаем и компилируем ядро оптимизатора
        kernels_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cl_kernels'))
        with open(os.path.join(kernels_dir, 'optimizer.cl'), 'r', encoding='utf-8') as f:
            self.program = JITProgramCache.get_or_compile(self.ctx, f.read())

    def zero_grad(self):
        """Очищает градиенты всех параметров перед новым шагом backward"""
        for param in self.parameters:
            if param.grad_buffer is not None:
                cl.enqueue_fill_buffer(self.queue, param.grad_buffer, np.float32(0.0), 0, param.nbytes)
        self.queue.finish()

    def step(self):
        """Выполняет in-place обновление весов во VRAM через Си-ядро"""
        for param in self.parameters:
            if param.grad_buffer is None:
                continue # Пропускаем параметры, для которых не считались градиенты
                
            # Вычисляем 2D-сетку потоков под размер текущего тензора
            total_elements = param.size
            total_count = int(np.ceil(total_elements / float(self.warp_size)) * self.warp_size)
            global_size = (total_count // self.warp_size, self.warp_size)
            
            # Запускаем Си-ядро обновления
            self.program.sgd_step_kernel(
                self.queue, global_size, None,
                param.data_buffer, param.grad_buffer,
                np.int32(total_elements), np.float32(self.lr), np.float32(self.weight_decay),
                np.int32(self.warp_size)
            )
            
        self.queue.finish() # Синхронизируем GPU после шага оптимизации


class AdamWOptimizer:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Оптимизатор AdamW, выполняющий все расчеты и хранение моментов строго во VRAM.
        """
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        
        if not self.parameters:
            raise ValueError("Список параметров для оптимизатора не должен быть пустым.")
            
        self.ctx = self.parameters[0].ctx
        self.queue = self.parameters[0].queue
        self.step_count = 0  # Счётчик итераций (t) для Bias Correction

        # Узнаем размер варпа
        dummy_code = "__kernel void dummy() {}"
        dummy_prg = cl.Program(self.ctx, dummy_code).build()
        try:
            self.warp_size = dummy_prg.dummy.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                self.queue.device
            )
        except Exception:
            self.warp_size = 32

        # Загружаем Си-код (он подтянет и SGD, и AdamW)
        kernels_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cl_kernels'))
        with open(os.path.join(kernels_dir, 'optimizer.cl'), 'r', encoding='utf-8') as f:
            self.program = JITProgramCache.get_or_compile(self.ctx, f.read())

        # ЛЕНИВОЕ ВЫДЕЛЕНИЕ ПАМЯТИ ПОД МОМЕНТЫ (m и v)
        # Каждому тензору параметров сопоставляем два чистых буфера OpenCL той же длины
        self.exp_avg = {}
        self.exp_avg_sq = {}
        
        for param in self.parameters:
            # Выделяем буферы во VRAM и заполняем их нулями
            m_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, param.nbytes)
            v_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, param.nbytes)
            cl.enqueue_fill_buffer(self.queue, m_buf, np.float32(0.0), 0, param.nbytes)
            cl.enqueue_fill_buffer(self.queue, v_buf, np.float32(0.0), 0, param.nbytes)
            
            self.exp_avg[param] = m_buf
            self.exp_avg_sq[param] = v_buf
            
        self.queue.finish()

    def zero_grad(self):
        """Обнуляет буферы градиентов параметров"""
        for param in self.parameters:
            if param.grad_buffer is not None:
                cl.enqueue_fill_buffer(self.queue, param.grad_buffer, np.float32(0.0), 0, param.nbytes)
        self.queue.finish()

    def step(self):
        """Шаг обновления параметров AdamW"""
        self.step_count += 1
        
        # Вычисляем Bias Correction на процессоре (один раз на шаг для экономии тактов GPU)
        bias_correction1 = float(1.0 - (self.beta1 ** self.step_count))
        bias_correction2 = float(1.0 - (self.beta2 ** self.step_count))

        for param in self.parameters:
            if param.grad_buffer is None:
                continue
                
            total_elements = param.size
            total_count = int(np.ceil(total_elements / float(self.warp_size)) * self.warp_size)
            global_size = (total_count // self.warp_size, self.warp_size)
            
            # Извлекаем скрытые буферы моментов для текущего тензора весов
            m_buf = self.exp_avg[param]
            v_buf = self.exp_avg_sq[param]

            # Вызываем Си-ядро AdamW
            self.program.adamw_step_kernel(
                self.queue, global_size, None,
                param.data_buffer, param.grad_buffer, m_buf, v_buf,
                np.int32(total_elements), np.float32(self.lr),
                np.float32(self.beta1), np.float32(self.beta2),
                np.float32(self.eps), np.float32(self.weight_decay),
                np.float32(bias_correction1), np.float32(bias_correction2),
                np.int32(self.warp_size)
            )
            
        self.queue.finish()
