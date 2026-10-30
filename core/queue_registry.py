# core/queue_registry.py

class DeviceQueueRegistry:
    """
    Статический глобальный регистратор активной очереди команд OpenCL (Паттерн Одиночка).
    Гарантирует, что runner_step_backward всегда получает живую и валидную очередь.
    """
    _current_queue = None

    @classmethod
    def set_current(cls, queue):
        """Устанавливает текущую активную очередь команд"""
        cls._current_queue = queue

    @classmethod
    def get_current(cls):
        """Возвращает актуальный валидный объект очереди"""
        if cls._current_queue is None:
            raise RuntimeError("Очередь OpenCL не зарегистрирована в DeviceQueueRegistry!")
        return cls._current_queue
