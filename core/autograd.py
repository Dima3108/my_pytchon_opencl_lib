import pyopencl as cl
IS_DEBUG=True
class AutogradContext:
    """
    Единый менеджер графа вычислений, привязанный к конкретной очереди OpenCL.
    Устраняет разрозненность стеков у разных тензоров.
    """
    """
    Глобальный менеджер графа вычислений (Синглтон), 
    привязанный к текущей активной очереди OpenCL.
    """
    _instance = None  # Статическая переменная для хранения единственного инстанса

    def __new__(cls, *args, **kwargs):
        # Если инстанс еще не создан, создаем его
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_current(cls):
        """Глобальный статический метод для получения активного контекста"""
        if cls._instance is None:
            raise RuntimeError("AutogradContext еще не был инициализирован. Сначала создайте его экземпляр!")
        return cls._instance

    def __init__(self, queue: cl.CommandQueue):
        # Инициализируем атрибуты только один раз при первом создании
        if not hasattr(self, '_initialized'):
            from core.queue_registry import DeviceQueueRegistry
            self.queue = DeviceQueueRegistry.get_current()
            self.tape = []
            self.is_training = True
            self._initialized = True

    def register_backward(self, backward_fn):
        """Добавляет операцию в общую ленту"""
        if self.is_training:
            self.tape.append(backward_fn)
            if IS_DEBUG:
                print("[autograd_debug] добавлен элемент в глобальный стек обратного прохода!")

    def run_backward(self):
        """Последовательно разворачивает всю ленту от конца к началу с защитой от рекурсии"""
        print(f"\n[autograd_debug] СТАРТ BACKWARD. Всего задач в стеке: {len(self.tape)}")
        
        # КРИТИЧЕСКИЙ ПРЕДОХРАНИТЕЛЬ: временно отключаем запись новых операций в режиме backward
        self.is_training = False 
        
        try:
            while self.tape:
                current_len = len(self.tape)
                backward_fn = self.tape.pop()
                backward_fn()
                
                if IS_DEBUG:
                    print(f"[autograd_debug] Итерация завершена! Задач в стеке: {len(self.tape)}")
        finally:
            # Возвращаем режим обучения обратно по завершении прохода
            self.is_training = True
            
        self.queue.finish()



    def zero_grad(self, params):
        """Безопасно зануляет буферы градиентов для списка тензоров"""
        for param in params:
            #param.requires_grad()
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
    Берет контекст напрямую из глобального Синглтона.
    """
    from core.tensor import Tensor
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1. Выполняем оригинальный метод (bmm, softmax и т.д.)
        out_tensor = func(self, *args, **kwargs)
        
        # Получаем глобальный синглтон контекста вычислений
        try:
            active_context = AutogradContext.get_current()
        except RuntimeError:
            return out_tensor # Если контекст не инициализирован (инференс), выходим
            
        if not active_context.is_training:
            return out_tensor

        # 2. Проверяем, требуют ли входные тензоры расчета градиентов
        input_tensors = [arg for arg in args if isinstance(arg, Tensor)]
        requires_grad = any(t.grad_buffer is not None for t in input_tensors)
        
        if requires_grad:
            out_tensor.require_grad()
            
            # Находим парный backward метод
            backward_method_name = f"{func.__name__}_backward"
            backward_method = getattr(self, backward_method_name, None)
            
            if backward_method is not None:
                def backward_closure():
                    backward_method(out_tensor, *args, **kwargs)
                
                # Записываем действие напрямую в синглтон!
                active_context.register_backward(backward_closure)
                
        return out_tensor
    return wrapper

