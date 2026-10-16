import os
import sys
import numpy as np
import pyopencl as cl
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from core.optimizer import SGDOptimizer, AdamWOptimizer
from compiler.host_returner import HostReturner

# ============================================================================
# ГЛОБАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ OPENCL (Один контекст на оба теста)
# ============================================================================
platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("OpenCL платформы не найдены.")
devices = platforms[0].get_devices()[0]
ctx = cl.Context([devices])
queue = cl.CommandQueue(ctx)

print(f"--- Инициализирован единый контекст GPU для тестов: {devices.name} ---")


def test_sgd_optimizer():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА: SGD OPTIMIZER IN VRAM")
    print("="*70)

    shape = (8, 8)
    np.random.seed(42)
    w_init = np.random.randn(*shape).astype(np.float32)
    g_init = np.random.randn(*shape).astype(np.float32)

    LR = 0.1
    WEIGHT_DECAY = 0.01

    # Передаем глобальные ctx и queue
    W_cl = Tensor.from_numpy(w_init, ctx, queue)
    W_cl.require_grad()
    cl.enqueue_copy(queue, W_cl.grad_buffer, g_init)
    queue.finish()

    optimizer_cl = SGDOptimizer([W_cl], lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_cl.step()

    cl_res_w = HostReturner.to_cpu(W_cl)

    t_W = torch.tensor(w_init, requires_grad=True)
    t_W.grad = torch.tensor(g_init)

    torch_optimizer = torch.optim.SGD([t_W], lr=LR, weight_decay=WEIGHT_DECAY)
    torch_optimizer.step()

    np.testing.assert_allclose(cl_res_w, t_W.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("  ✅ SGD Step Forward: Веса во VRAM обновились со 100% точностью!")

    optimizer_cl.zero_grad()
    cl_res_g = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(queue, cl_res_g, W_cl.grad_buffer)
    queue.finish()

    assert np.all(cl_res_g == 0.0), "Ошибка: Метод zero_grad не обнулил буфер градиентов!"
    print("  ✅ SGD Zero Grad: Буферы градиентов успешно очищены нулями!")


def test_adamw_optimizer():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА: ADAMW OPTIMIZER IN VRAM")
    print("="*70)

    shape = (16, 16)
    np.random.seed(123)
    w_init = np.random.randn(*shape).astype(np.float32)
    g_init = np.random.randn(*shape).astype(np.float32)

    LR = 0.002
    WD = 0.05
    BETAS = (0.9, 0.95)
    EPS = 1e-7

    # ИСПРАВЛЕНИЕ: Используем ГЛОБАЛЬНЫЕ ctx и queue, исключая INVALID_MEM_OBJECT!
    W_cl = Tensor.from_numpy(w_init, ctx, queue)
    W_cl.require_grad()
    cl.enqueue_copy(queue, W_cl.grad_buffer, g_init)
    queue.finish()

    optimizer_cl = AdamWOptimizer([W_cl], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    
    optimizer_cl.step()
    
    g_init_2 = np.random.randn(*shape).astype(np.float32)
    cl.enqueue_copy(queue, W_cl.grad_buffer, g_init_2)
    queue.finish()
    optimizer_cl.step()

    cl_res_w = HostReturner.to_cpu(W_cl)

    t_W = torch.tensor(w_init, requires_grad=True)
    t_W.grad = torch.tensor(g_init)
    
    torch_optimizer = torch.optim.AdamW([t_W], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    torch_optimizer.step()
    
    torch_optimizer.zero_grad()
    t_W.grad = torch.tensor(g_init_2)
    torch_optimizer.step()

    np.testing.assert_allclose(cl_res_w, t_W.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ AdamW Step Forward: Моменты и веса обновились идеально точно!")
    print("\n🎉 ТЕСТ ОПТИМИЗАТОРА ADAMW УСПЕШНО ПРОЙДЕН!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_sgd_optimizer()
    test_adamw_optimizer()


