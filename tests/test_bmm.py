import os
import numpy as np
import pyopencl as cl
import torch

def test_bmm_tiled():
    # 1. Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    print(f"\n[Тест BMM] Проверка Пакетного Matmul (bmm_tiled) на: {device.name}")

    # 2. Задаем размерности пакета (Batch, M, K, N)
    # Имитируем умножение Q [B, H, L, D_head] на K_T [B, H, D_head, L] в Attention
    BATCH = 4       # Например, Batch * Num_Heads
    M, K, N = 32, 64, 32

    batch_stride_A = M * K
    batch_stride_B = K * N
    batch_stride_C = M * N

    # 3. Генерируем случайные данные на Host (CPU)
    A_host = np.random.randn(BATCH, M, K).astype(np.float32)
    B_host = np.random.randn(BATCH, K, N).astype(np.float32)
    C_host = np.zeros((BATCH, M, N), dtype=np.float32)

    # 4. Читаем и компилируем ядро bmm_tiled
    kernel_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'matmul.cl')
    with open(kernel_path, 'r', encoding='utf-8') as f:
        kernel_source = f.read()
    program = cl.Program(ctx, kernel_source).build()

    # 5. Выделяем OpenCL буферы в VRAM
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY, C_host.nbytes)

    # 6. Настройка сетки потоков (Global и Local Size)
    # Для bmm_tiled мы используем ТРЕХМЕРНУЮ сетку потоков (3D Grid)
    # local_size настраиваем по X и Y под TILE_SIZE=16, Z (батч) равен 1
    local_size = (16, 16, 1)

    # Округляем глобальные размеры по X и Y вверх до кратных 16
    global_size_x = int(np.ceil(N / 16.0) * 16)
    global_size_y = int(np.ceil(M / 16.0) * 16)
    global_size_z = BATCH
    global_size = (global_size_x, global_size_y, global_size_z)

    # 7. Запуск ядра
    program.bmm_tiled(
        queue, global_size, local_size,
        a_g, b_g, c_g,
        np.int32(M), np.int32(K), np.int32(N),
        np.int32(batch_stride_A), np.int32(batch_stride_B), np.int32(batch_stride_C)
    )

    # 8. Считываем результат обратно в память CPU
    cl.enqueue_copy(queue, C_host, c_g)

    # 9. Считаем эталон через PyTorch BMM
    A_torch = torch.from_numpy(A_host)
    B_torch = torch.from_numpy(B_host)
    C_torch = torch.bmm(A_torch, B_torch).numpy()

    # 10. Проверка на совпадение результатов с точностью float32
    try:
        np.testing.assert_allclose(C_host, C_torch, rtol=1e-4, atol=1e-4)
        print("✅ Успешно! Пакетное матричное умножение bmm_tiled работает корректно и совпадает с PyTorch.")
    except AssertionError as e:
        print("❌ Ошибка верификации BMM:")
        print(e)

if __name__ == "__main__":
    test_bmm_tiled()
