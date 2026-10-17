from core.tensor import Tensor


from core.autograd import AutogradContext

class Module:
    def __init__(self, ctx, queue):
        self._modules = {}
        self.ctx = ctx
        self.queue = queue
        # Создаем единый контекст автодифференцирования для всего модуля
        self.autograd_ctx = AutogradContext(queue)
        self.training = True

    def train(self, mode: bool = True):
        """Переключает режим обучения для модели, подмодулей и автограда"""
        self.training = mode
        self.autograd_ctx.is_training = mode
        
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train(mode)
                
        if hasattr(self, 'runner') and self.runner is not None:
            self.runner.is_training = mode


    def __setattr__(self, name, value):
        """
        Автоматически перехватывает объявление подмодулей (например, self.linear = Linear(...))
        и сохраняет ссылки на них в системный словарь _modules.
        """
        if isinstance(value, Module) or hasattr(value, 'forward'):
            self.__dict__['_modules'][name] = value
        super().__setattr__(name, value)



    def eval(self):
        """Переключает модель в режим инференса (тестирования)"""
        self.train(mode=False)

    def parameters(self):
        """
        Рекурсивно обходит всю модель, собирает и возвращает генератор 
        всех обучаемых параметров (объектов Tensor, у которых включен require_grad)
        """
        params = []
        
        # 1. Ищем тензоры внутри текущего модуля
        for name, attr in self.__dict__.items():
            if isinstance(attr, Tensor) and attr.grad_buffer is not None:
                if attr not in params:
                    params.append(attr)
                    
        # 2. Рекурсивно лезем во все дочерние подмодули
        for name, module in self._modules.items():
            # Если это кастомный слой (например, Linear или Embedding), вытаскиваем его веса
            for attr_name, attr_val in module.__dict__.items():
                if isinstance(attr_val, Tensor):
                    # Если у тензора лениво не выделен grad_buffer, но это обучаемый вес
                    if attr_name in ['weight', 'bias']:
                        attr_val.require_grad()
                    if attr_val not in params:
                        params.append(attr_val)
                        
            # Если дочерний модуль сам унаследован от Module, вызываем его параметры
            if isinstance(module, Module):
                for p in module.parameters():
                    if p not in params:
                        params.append(p)
                        
        return params

    def __call__(self, *args, **kwargs):
        """Позволяет вызывать модуль как функцию: model(x) вместо model.forward(x)"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Каждый модуль обязан реализовать свой метод forward()!")
