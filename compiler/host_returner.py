import numpy as np
import pyopencl as cl

class HostReturner:
    @staticmethod
    def to_cpu(tensor):
        """
        Скачивает данные тензора из OpenCL VRAM в RAM CPU 
        и возвращает его как структурированный массив NumPy.
        """
        # Создаем пустой массив нужной формы в оперативной памяти CPU
        host_array = np.empty(tensor.shape, dtype=tensor.dtype)
        
        # Безопасно копируем данные из буфера OpenCL
        cl.enqueue_copy(tensor.queue, host_array, tensor.data_buffer)
        
        # Важнейший шаг: встроенная синхронизация очереди OpenCL.
        # Гарантирует, что все вычисления на GPU завершены перед тем, как Python прочитает массив.
        tensor.queue.finish()
        
        return host_array

    @staticmethod
    def to_cpu_grad(tensor):
        """Скачивает градиенты тензора из OpenCL VRAM в RAM CPU"""
        if tensor.grad_buffer is None:
            raise RuntimeError(f"Тензор {tensor} не имеет буфера градиентов (require_grad не вызывался).")
            
        host_grad = np.empty(tensor.shape, dtype=tensor.dtype)
        cl.enqueue_copy(tensor.queue, host_grad, tensor.grad_buffer)
        tensor.queue.finish()
        
        return host_grad

    @classmethod
    def print_tensor(cls, tensor, name="Tensor", print_grad=False):
        """
        Красиво выводит тензор (и/или его градиенты) в консоль для отладки, 
        имитируя форматирование PyTorch.
        """
        print("=" * 60)
        print(f"📊 ОТЛАДКА ТЕНЗОРА: {name}")
        print(f"  • Форма (shape): {tensor.shape}")
        print(f"  • Всего элементов: {tensor.size}")
        print(f"  • Тип данных: {tensor.dtype.name}")
        print("-" * 60)
        
        # Скачиваем данные
        data = cls.to_cpu(tensor)
        
        # Настраиваем NumPy на компактный вывод (чтобы не спамить консоль миллионами чисел)
        with np.printoptions(precision=4, suppress=True, edgeitems=3, linewidth=75):
            print("Данные (Forward):")
            print(data)
            
            if print_grad:
                print("\nГрадиенты (Backward):")
                if tensor.grad_buffer is not None:
                    grad = cls.to_cpu_grad(tensor)
                    print(grad)
                else:
                    print("  [Градиенты отсутствуют / require_grad=False]")
                    
        print("=" * 60 + "\n")
