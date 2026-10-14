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


def test_bmm_glob_pipeline():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ГЛОБАЛЬНОГО ТЕСТА: BMM ПАКЕТНОЕ УМНОЖЕНИЕ (FORWARD + BACKWARD)")
    print("="*70)
    print(f"Тестирование на устройстве: {devices[0].name}")

    # Задаем размерности пакета матриц (Batch, M, K, N)
    # Имитируем реальный срез MHA: 4 головы, проекция 32x64 и 64x48
    BATCH, M, K, N = 4, 32, 64, 48

    np.random.seed(777)
    A_init = np.random.randn(BATCH, M, K).astype(np.float32)
    B_init = np.random.randn(BATCH, K, N).astype(np.float32)

    # Создаем тензоры в нашей библиотеке
    A_cl = Tensor.from_numpy(A_init, ctx, queue)
    B_cl = Tensor.from_numpy(B_init, ctx, queue)

    print("\nВычисление BMM Forward на GPU...")
    C_cl = nnclglob.bmm(A_cl, B_cl)

    # Генерируем случайный входящий градиент ошибок grad_output
    grad_output_init = np.random.randn(BATCH, M, N).astype(np.float32)
    grad_output_cl = Tensor.from_numpy(grad_output_init, ctx, queue)

    print("Вычисление BMM Backward на GPU...")
    grad_A_cl, grad_B_cl = nnclglob.bmm_backward(grad_output_cl, A_cl, B_cl)

    # Скачиваем результаты расчетов из VRAM через HostReturner
    cl_res_c = HostReturner.to_cpu(C_cl)
    cl_res_grad_A = HostReturner.to_cpu(grad_A_cl)
    cl_res_grad_B = HostReturner.to_cpu(grad_B_cl)

    # ЭТАЛОННЫЙ РАСЧЕТ НА PYTORCH
    print("\nВерификация расчетов с оригинальным PyTorch...")
    t_A = torch.tensor(A_init, requires_grad=True)
    t_B = torch.tensor(B_init, requires_grad=True)
    
    # Прямой ход PyTorch
    t_C = torch.bmm(t_A, t_B)
    
    # Обратный ход PyTorch
    t_grad_output = torch.tensor(grad_output_init)
    t_C.backward(t_grad_output)

    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # Проверяем Forward
    np.testing.assert_allclose(cl_res_c, t_C.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Forward BMM: Полное совпадение перемноженных матриц пакета!")

    # Проверяем Backward по матрице A
    np.testing.assert_allclose(cl_res_grad_A, t_A.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad A): Градиенты левой матрицы полностью совпадают!")

    # Проверяем Backward по матрице B
    np.testing.assert_allclose(cl_res_grad_B, t_B.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad B): Градиенты правой матрицы полностью совпадают!")

    print("\n🎉 ТЕСТ ПАКЕТНОГО МАТРИЧНОГО УМНОЖЕНИЯ УСПЕШНО ПРОЙДЕН!")
    print("="*70 + "\n")


import torch.nn.functional as F

def test_causal_conv1d_glob_pipeline():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ГЛОБАЛЬНОГО ТЕСТА: CAUSAL CONV1D (FORWARD + BACKWARD)")
    print("="*70)
    print(f"Тестирование на устройстве: {devices[0].name}")

    Batch, Channels, Length = 2, 8, 16
    kernel_size = 4

    np.random.seed(42)
    X_init = np.random.randn(Batch, Channels, Length).astype(np.float32)
    W_init = np.random.randn(Channels, kernel_size).astype(np.float32)
    B_init = np.random.randn(Channels).astype(np.float32)

    # Создаем тензоры в OpenCL
    x_cl = Tensor.from_numpy(X_init, ctx, queue)
    w_cl = Tensor.from_numpy(W_init, ctx, queue)
    b_cl = Tensor.from_numpy(B_init, ctx, queue)

    print("\nВычисление CausalConv1d Forward на GPU...")
    y_cl = nnclglob.causal_conv1d(x_cl, w_cl, b_cl)

    # Входящий градиент ошибок
    grad_output_init = np.random.randn(Batch, Channels, Length).astype(np.float32)
    grad_output_cl = Tensor.from_numpy(grad_output_init, ctx, queue)

    print("Вычисление CausalConv1d Backward на GPU...")
    grad_X_cl, grad_W_cl, grad_B_cl = nnclglob.causal_conv1d_backward(grad_output_cl, x_cl, w_cl)

    # Выкачиваем результаты
    cl_res_y = HostReturner.to_cpu(y_cl)
    cl_res_grad_X = HostReturner.to_cpu(grad_X_cl)
    cl_res_grad_W = HostReturner.to_cpu(grad_W_cl)
    cl_res_grad_B = HostReturner.to_cpu(grad_B_cl)

    # 6. ЭТАЛОННЫЙ РАСЧЕТ НА PYTORCH
    print("\nВерификация расчетов с оригинальным PyTorch...")
    t_X = torch.tensor(X_init, requires_grad=True)
    t_B = torch.tensor(B_init, requires_grad=True)
    
    # ИСПРАВЛЕНИЕ: Сначала делаем структурное изменение unsqueeze, 
    # и только ПОТОМ помечаем тензор как лист для автограда через .requires_grad_()
    t_W = torch.tensor(W_init).unsqueeze(1).requires_grad_()

    # Прямой ход PyTorch с каузальным паддингом
    x_padded = F.pad(t_X, (kernel_size - 1, 0))
    t_Y = F.conv1d(x_padded, t_W, bias=t_B, groups=Channels)

    # Обратный ход PyTorch
    t_grad_output = torch.tensor(grad_output_init)
    t_Y.backward(t_grad_output)


    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    np.testing.assert_allclose(cl_res_y, t_Y.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Forward Conv1d: Полное совпадение карт признаков свертки!")

    np.testing.assert_allclose(cl_res_grad_X, t_X.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad X): Градиенты по входному сигналу полностью совпали!")

    np.testing.assert_allclose(cl_res_grad_W, t_W.grad.squeeze(1).numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad Weight): Градиенты по весам фильтров полностью совпали!")

    np.testing.assert_allclose(cl_res_grad_B, t_B.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Backward (Grad Bias): Градиенты по смещениям полностью совпали!")

    print("\n🎉 ТЕСТ КАУЗАЛЬНОЙ СВЕРТКИ УСПЕШНО ПРОЙДЕН!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_rmsnorm_glob_pipeline()
    test_softmax_glob_pipeline()
    test_bmm_glob_pipeline()
    test_causal_conv1d_glob_pipeline() # <-- Добавили новый шаг




