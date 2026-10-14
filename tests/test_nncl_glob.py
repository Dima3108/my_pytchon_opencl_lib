import os
import sys
import numpy as np
import pyopencl as cl
import torch

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from compiler.nncl_glob import NnClGlob
from compiler.host_returner import HostReturner

# ============================================================================
# ГЛОБАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ OPENCL (Один контекст на весь запуск)
# ============================================================================
platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("OpenCL платформы не найдены.")
devices = platforms[0].get_devices()
ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)

print(f"--- Инициализирован глобальный контекст GPU: {devices[0].name} ---")
nnclglob = NnClGlob(ctx, queue)


def test_rmsnorm_glob_pipeline():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ГЛОБАЛЬНОГО ТЕСТА: RMSNORM (FORWARD + BACKWARD)")
    print("="*70)

    B, L, D = 2, 4, 16
    shape = (B * L, D)

    np.random.seed(42)
    x_init = np.random.randn(*shape).astype(np.float32)
    gamma_init = np.random.uniform(0.5, 1.5, size=(D,)).astype(np.float32)

    # Используем глобальные ctx и queue
    x_cl = Tensor.from_numpy(x_init, ctx, queue)
    gamma_cl = Tensor.from_numpy(gamma_init, ctx, queue)

    print("\nВычисление RMSNorm Forward на GPU...")
    y_cl = nnclglob.rmsnorm(x_cl, gamma_cl)

    grad_output_init = np.random.randn(*shape).astype(np.float32)
    grad_output_cl = Tensor.from_numpy(grad_output_init, ctx, queue)
    grad_output_cl._inv_rms_cache = y_cl._inv_rms_cache

    print("Вычисление RMSNorm Backward на GPU...")
    grad_input_cl, grad_gamma_cl = nnclglob.rmsnorm_backward(grad_output_cl, x_cl, gamma_cl)

    cl_res_y = HostReturner.to_cpu(y_cl)
    cl_res_grad_in = HostReturner.to_cpu(grad_input_cl)
    cl_res_grad_gamma = HostReturner.to_cpu(grad_gamma_cl)

    print("\nВерификация расчетов с оригинальным PyTorch...")
    t_x = torch.tensor(x_init, requires_grad=True)
    torch_rmsnorm = torch.nn.RMSNorm(D)
    torch_rmsnorm.weight.data.copy_(torch.tensor(gamma_init))
    t_y = torch_rmsnorm(t_x)
    
    t_grad_output = torch.tensor(grad_output_init)
    t_y.backward(t_grad_output)
    pytorch_grad_gamma = torch_rmsnorm.weight.grad.numpy()

    np.testing.assert_allclose(cl_res_y, t_y.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Forward: Полное совпадение нормализованных данных!")

    np.testing.assert_allclose(cl_res_grad_in, t_x.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad Input): Полное совпадение градиентов по входу X!")

    np.testing.assert_allclose(cl_res_grad_gamma, pytorch_grad_gamma, rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad Gamma): Полное совпадение градиентов по весам слоёв!")


def test_softmax_glob_pipeline():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ГЛОБАЛЬНОГО ТЕСТА: SOFTMAX (FORWARD + BACKWARD)")
    print("="*70)

    ROWS, SEQ_LEN = 16, 32
    shape = (ROWS, SEQ_LEN)

    np.random.seed(123)
    x_init = np.random.randn(*shape).astype(np.float32)

    # Используем глобальные ctx и queue
    x_cl = Tensor.from_numpy(x_init, ctx, queue)

    print("\nВычисление Softmax Forward на GPU...")
    y_cl = nnclglob.softmax(x_cl)

    grad_output_init = np.random.randn(*shape).astype(np.float32)
    grad_output_cl = Tensor.from_numpy(grad_output_init, ctx, queue)

    print("Вычисление Softmax Backward на GPU...")
    grad_input_cl = nnclglob.softmax_backward(grad_output_cl, y_cl)

    cl_res_y = HostReturner.to_cpu(y_cl)
    cl_res_grad_in = HostReturner.to_cpu(grad_input_cl)

    print("\nВерификация расчетов с оригинальным PyTorch...")
    t_x = torch.tensor(x_init, requires_grad=True)
    t_y = torch.softmax(t_x, dim=-1)
    
    t_grad_output = torch.tensor(grad_output_init)
    t_y.backward(t_grad_output)

    np.testing.assert_allclose(cl_res_y, t_y.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Forward: Полное совпадение распределения вероятностей Softmax!")

    np.testing.assert_allclose(cl_res_grad_in, t_x.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad Input): Градиенты Softmax полностью совпадают с PyTorch!")


if __name__ == "__main__":
    test_rmsnorm_glob_pipeline()
    test_softmax_glob_pipeline()


