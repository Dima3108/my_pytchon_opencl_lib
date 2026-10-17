import pyopencl as cl

class AutogradContext:
    """
    Единый менеджер графа вычислений, привязанный к конкретной очереди OpenCL.
    Устраняет разрозненность стеков у разных тензоров.
    """
    def __init__(self, queue: cl.CommandQueue):
        self.queue = queue
        self.tape = []          # Единая лента (стек) для backward-колбэков
        self.is_training = True # Флаг режима обучения (запись в tape)

    def register_backward(self, backward_fn):
        """Добавляет операцию в общую ленту, если включен режим обучения"""
        if self.is_training:
            self.tape.append(backward_fn)

    def run_backward(self):
        """Последовательно разворачивает всю ленту от конца к началу"""
        while self.tape:
            backward_fn = self.tape.pop()
            backward_fn()
        # Гарантируем, что все backward-ядра на GPU завершили работу
        self.queue.finish()

    def zero_grad(self, params):
        """Безопасно зануляет буферы градиентов для списка тензоров"""
        for param in params:
            if param.grad_buffer is not None:
                cl.enqueue_fill_buffer(
                    self.queue, 
                    param.grad_buffer, 
                    param.dtype.type(0.0), 
                    0, 
                    param.nbytes
                )
        self.queue.finish()


import functools
from core.tensor import Tensor

def auto_backward(func):
    """
    Декоратор для автоматической регистрации графа.
    Ищет парный метод с суффиксом '_backward' и добавляет его в стек контекста.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1. Выполняем оригинальный метод (например, bmm или softmax)
        out_tensor = func(self, *args, **kwargs)
        
        # Если autograd отключен в модели (например, при eval/inference), просто возвращаем результат
        if hasattr(self, 'autograd_ctx') and not self.autograd_ctx.is_training:
            return out_tensor

        # 2. Проверяем, требуют ли входные тензоры расчета градиентов
        input_tensors = [arg for arg in args if isinstance(arg, Tensor)]
        requires_grad = any(t.grad_buffer is not None for t in input_tensors)
        
        if requires_grad:
            # Выделяем буфер градиента для выходного тензора, если его нет
            out_tensor.require_grad()
            
            # Находим парный backward метод по имени (например, для 'bmm' -> 'bmm_backward')
            backward_method_name = f"{func.__name__}_backward"
            backward_method = getattr(self, backward_method_name, None)
            
            if backward_method is not None:
                # Создаем замыкание (колбэк), которое будет вызвано во время run_backward()
                def backward_closure():
                    # Автоматически вызываем: метод_backward(grad_output, *входные_тензоры)
                    backward_method(out_tensor, *args, **kwargs)
                
                # Записываем это действие в единую очередь нашей модели!
                if hasattr(self, 'autograd_ctx'):
                    self.autograd_ctx.register_backward(backward_closure)
                    
        return out_tensor
    return wrapper
