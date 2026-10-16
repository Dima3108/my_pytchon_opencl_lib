import os
import sys
import numpy as np
import pyopencl as cl
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from core.optimizer import SGDOptimizer
from compiler.host_returner import HostReturner

def test_sgd_optimizer():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА: SGD OPTIMIZER IN VRAM")
    print("="*70)

    # Инициализация OpenCL
    platforms = cl.get_platforms()[0]
    devices = platforms.get_devices()[0]
    ctx = cl.Context([devices])
    queue = cl.CommandQueue(ctx)
    print(f"Тестирование оптимизатора на устройстве: {devices.name}")

    # Имитируем матрицу весов слоя (например, 64 элемента)
    shape = (8, 8)
    np.random.seed(42)
    w_init = np.random.randn(*shape).astype(np.float32)
    g_init = np.random.randn(*shape).astype(np.float32)

    # Настройка параметров обучения
    LR = 0.1
    WEIGHT_DECAY = 0.01

    # 1. Наш Оптимизатор
    W_cl = Tensor.from_numpy(w_init, ctx, queue)
    W_cl.require_grad()
    # Загружаем имитированный градиент в grad_buffer
    cl.enqueue_copy(queue, W_cl.grad_buffer, g_init)
    queue.finish()

    optimizer_cl = SGDOptimizer([W_cl], lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_cl.step()

    cl_res_w = HostReturner.to_cpu(W_cl)

    # 2. Эталон в PyTorch
    t_W = torch.tensor(w_init, requires_grad=True)
    # Принудительно подставляем такой же градиент в PyTorch
    t_W.grad = torch.tensor(g_init)

    torch_optimizer = torch.optim.SGD([t_W], lr=LR, weight_decay=WEIGHT_DECAY)
    torch_optimizer.step()

    # Сравнение результатов
    np.testing.assert_allclose(cl_res_w, t_W.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("  ✅ SGD Step Forward: Веса во VRAM обновились со 100% точностью!")

    # Проверка zero_grad
    optimizer_cl.zero_grad()
    cl_res_g = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(queue, cl_res_g, W_cl.grad_buffer)
    queue.finish()

    assert np.all(cl_res_g == 0.0), "Ошибка: Метод zero_grad не обнулил буфер градиентов!"
    print("  ✅ SGD Zero Grad: Буферы градиентов успешно очищены нулями!")

    print("\n🎉 ТЕСТ ОПТИМИЗАТОРА SGD УСПЕШНО ПРОЙДЕН!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_sgd_optimizer()
