import os
import sys
import numpy as np
import pyopencl as cl
import torch

# Добавляем корневую директорию проекта в путь поиска модулей Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from compiler.nncl_compiler import LocalRunner
from compiler.host_returner import HostReturner

class DummyNnclLoc:
    def sin(self, x): return x
nnclloc = DummyNnclLoc()

def test_heterogeneous_dynamic_sizes():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК КОМПЛЕКСНОГО ТЕСТА ДИНАМИЧЕСКИХ РАЗМЕРОВ И СКАЛЯРОВ")
    print("="*70)

    # Инициализация единого OpenCL окружения
    platforms = cl.get_platforms()[0]
    devices = platforms.get_devices()[0]
    ctx = cl.Context([devices])
    queue = cl.CommandQueue(ctx)
    print(f"Тестирование динамических границ на устройстве: {devices.name}")

    SIZE_X = 7000
    SIZE_Y = 3400
    q = 0.23 

    # ИСПРАВЛЕНИЕ 1: Фиксируем сид строго ДО генерации всех массивов!
    np.random.seed(42)
    init_X = np.random.uniform(-1.0, 1.0, size=(SIZE_X,)).astype(np.float32)
    init_Y = np.random.uniform(-1.0, 1.0, size=(SIZE_Y,)).astype(np.float32)

    X = Tensor.from_numpy(init_X, ctx, queue)
    Y = Tensor.from_numpy(init_Y, ctx, queue)

    X.require_grad()
    Y.require_grad()

    runner = LocalRunner(ctx, queue, is_training=True)

    # ИСПРАВЛЕНИЕ 2: Объединяем цепочки вычислений в единые монолитные вызовы,
    # чтобы JIT-компилятор построил правильные связанные уравнения Си-кода!
    # 3. ВЫПОЛНЕНИЕ ПРЯМОГО ПРОХОДА (Многострочный Текстовый JIT-Блок)
    # Передаем всю последовательность действий как единую текстовую строку!
    print(f"\nВыполнение прямого прохода на GPU (Макс. сетка потоков: {SIZE_X})...")
    runner(f"""
        X = nnclloc.sin(X)
        Y = nnclloc.sin(Y)
        X = X - q
        Y = Y - q
    """)

    # ИМИТАЦИЯ ОБРАТНОГО ПРОХОДА (Backward)
    init_grad_X_output = np.ones((SIZE_X,), dtype=np.float32)
    init_grad_Y_output = np.ones((SIZE_Y,), dtype=np.float32)

    cl.enqueue_copy(queue, X.grad_buffer, init_grad_X_output)
    cl.enqueue_copy(queue, Y.grad_buffer, init_grad_Y_output)
    
    # Очищаем перед тестом
    cl.enqueue_fill_buffer(queue, X.grad_buffer, np.float32(0.0), 0, X.nbytes)
    cl.enqueue_fill_buffer(queue, Y.grad_buffer, np.float32(0.0), 0, Y.nbytes)
    queue.finish()

    # Сначала зальем входящий градиент (единицы) в буферы таргетов заново, 
    # так как они будут upstream-градиентами для цепочки
    cl.enqueue_copy(queue, X.grad_buffer, init_grad_X_output)
    cl.enqueue_copy(queue, Y.grad_buffer, init_grad_Y_output)
    queue.finish()

    print("Выполнение автоматического обратного прохода градиентов...")
    runner.run_backward()

    # Скачиваем результаты
    cl_res_X = HostReturner.to_cpu(X)
    cl_res_Y = HostReturner.to_cpu(Y)
    cl_grad_X = HostReturner.to_cpu_grad(X)
    cl_grad_Y = HostReturner.to_cpu_grad(Y)

    # ЭТАЛОННЫЙ РАСЧЕТ НА PYTORCH
    print("\nВерификация расчетов с оригинальным PyTorch...")
    t_X = torch.tensor(init_X, requires_grad=True)
    t_Y = torch.tensor(init_Y, requires_grad=True)

    # Точный повтор цепочки на PyTorch
    t_X_fused = torch.sin(t_X) - q
    t_Y_fused = torch.sin(t_Y) - q

    # Запускаем раздельные backward-контуры с единицами (как пришло сверху)
    t_X_fused.backward(torch.ones((SIZE_X,)))
    t_Y_fused.backward(torch.ones((SIZE_Y,)))

    # СРАВНЕНИЕ (ASSERT_ALLCLOSE)
    np.testing.assert_allclose(cl_res_X, t_X_fused.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cl_res_Y, t_Y_fused.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Forward (Динамические размеры): Данные X и Y полностью совпали с PyTorch!")

    np.testing.assert_allclose(cl_grad_X, t_X.grad.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cl_grad_Y, t_Y.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Защита границ): Все градиенты гетерогенных длин рассчитаны верно!")

    print("\n🎉 КОМПЛЕКСНЫЙ ТЕСТ ДИНАМИЧЕСКИХ ГРАНИЦ УСПЕШНО ПРОЙДЕН!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_heterogeneous_dynamic_sizes()

