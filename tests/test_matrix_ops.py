import os
import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

# Функция для компиляции и вызова нашего OpenCL elementwise ядра
def run_opencl_elementwise(ctx, queue, program, A_np, B_np, op_type=1):
    # op_type: 0 - сложение (+), 1 - умножение (*)
    
    # Дополняем размерности слева единицами до 4D для унификации в тесте
    max_dims = 4
    a_shape = list(A_np.shape)
    b_shape = list(B_np.shape)
    
    while len(a_shape) < max_dims: a_shape.insert(0, 1)
    while len(b_shape) < max_dims: b_shape.insert(0, 1)
    
    # Вычисляем выходную форму (правило бродкаста)
    c_shape = [max(a, b) for a, b in zip(a_shape, b_shape)]
    total_elements = int(np.prod(c_shape))
    
    C_np = np.zeros(c_shape, dtype=np.float32)
    
    # Превращаем формы в C-массивы int32
    a_dims = np.array(a_shape, dtype=np.int32)
    b_dims = np.array(b_shape, dtype=np.int32)
    c_dims = np.array(c_shape, dtype=np.int32)
    
    # Выделяем буферы
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_np.astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_np.astype(np.float32))
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY, C_np.nbytes)
    a_dims_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_dims)
    b_dims_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_dims)
    c_dims_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_dims)
    
    # Запуск ядра (каждый поток считает 1 элемент выходного тензора)
    program.elementwise_op(
        queue, (total_elements,), None,
        a_g, b_g, c_g,
        a_dims_g, b_dims_g, c_dims_g,
        np.int32(max_dims), np.int32(total_elements), np.int32(op_type)
    )
    
    cl.enqueue_copy(queue, C_np, c_g)
    return C_np

def test_all_matrix_operations():
    # 1. Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    print(f"--- Тестирование на устройстве: {device.name} ---")
    
    # Читаем ядро поэлементных операций (Broadcasting)
    kernel_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'elementwise_ops.cl')
    with open(kernel_path, 'r', encoding='utf-8') as f:
        kernel_source = f.read()
    program = cl.Program(ctx, kernel_source).build()

    # =========================================================================
    # ТЕСТ 1: Поэлементное умножение (*) с жестким Broadcasting
    # Ситуация: [B, H, L, D] * [1, 1, 1, D] (типично для RoPE или маскирования)
    # =========================================================================
    print("\n[Тест 1] Проверка Поэлементного умножения (*) с Broadcasting...")
    A_shape = (2, 4, 16, 64)
    B_shape = (1, 1, 1, 64)
    
    A_host = np.random.rand(*A_shape).astype(np.float32)
    B_host = np.random.rand(*B_shape).astype(np.float32)
    
    # Считаем через наше OpenCL ядро
    res_cl = run_opencl_elementwise(ctx, queue, program, A_host, B_host, op_type=1)
    
    # Считаем через эталонный PyTorch CPU
    res_torch = (torch.from_numpy(A_host) * torch.from_numpy(B_host)).numpy()
    
    np.testing.assert_allclose(res_cl, res_torch, rtol=1e-5, atol=1e-5)
    print("✅ Успешно! Broadcasting операции (*) работает стабильно и без падений.")

    # =========================================================================
    # ТЕСТ 2: Матричное умножение (@ / matmul) через Локальную Память
    # =========================================================================
    print("\n[Тест 2] Проверка Блочного Matmul (@)...")
    M, K, N = 64, 128, 96  # Размеры матриц
    
    # Генерируем тестовые матрицы
    A_host = np.random.randn(M, K).astype(np.float32)
    B_host = np.random.randn(K, N).astype(np.float32)
    C_host = np.zeros((M, N), dtype=np.float32)
    
    # Читаем и компилируем matmul.cl
    matmul_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'matmul.cl')
    with open(matmul_path, 'r', encoding='utf-8') as f:
        matmul_source = f.read()
    program_matmul = cl.Program(ctx, matmul_source).build()
    
    # Создаем OpenCL буферы
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY, C_host.nbytes)
    
    # Настройка сетки потоков (Global и Local Size)
    # Размер рабочей группы (Local Size) жестко фиксируем под 16x16, как определено в TILE_SIZE ядра
    local_size = (16, 16)
    
    # Округляем глобальный размер вверх, чтобы он был кратен 16 (размеру тайла)
    global_size_x = int(np.ceil(N / 16.0) * 16)
    global_size_y = int(np.ceil(M / 16.0) * 16)
    global_size = (global_size_x, global_size_y)
    
    # Запуск ядра
    program_matmul.matmul_tiled(
        queue, global_size, local_size,
        a_g, b_g, c_g,
        np.int32(M), np.int32(K), np.int32(N)
    )
    
    # Копируем результат обратно
    cl.enqueue_copy(queue, C_host, c_g)
    
    # Эталонное вычисление через PyTorch CPU
    res_torch = torch.matmul(torch.from_numpy(A_host), torch.from_numpy(B_host)).numpy()
    
    # Сверяем с допуском на точность float32
    np.testing.assert_allclose(C_host, res_torch, rtol=1e-4, atol=1e-4)
    print("✅ Успешно! Блочное ядро Matmul через локальную память работает корректно.")


    # =========================================================================
    # ТЕСТ 3: Свертка Conv1d с каузальным паддингом
    # =========================================================================
    print("\n[Тест 3] Проверка Каузального Depthwise Conv1d...")
    Batch, Channels, Length = 2, 64, 32
    kernel_size = 4

    # Генерируем случайные данные и веса
    X_host = np.random.randn(Batch, Channels, Length).astype(np.float32)
    W_host = np.random.randn(Channels, kernel_size).astype(np.float32)
    B_host = np.random.randn(Channels).astype(np.float32)
    Y_host = np.zeros_like(X_host)

    # Читаем и компилируем CausalConv1d.cl
    conv_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'CausalConv1d.cl')
    with open(conv_path, 'r', encoding='utf-8') as f:
        conv_source = f.read()
    program_conv = cl.Program(ctx, conv_source).build()

    # Буферы OpenCL
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_host)
    w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_host)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
    y_g = cl.Buffer(ctx, mf.WRITE_ONLY, Y_host.nbytes)

    # Запуск 3D сетки потоков: по времени, по каналам, по батчу
    global_size = (Length, Channels, Batch)

    program_conv.causal_conv1d_depthwise(
        queue, global_size, None,
        x_g, w_g, b_g, y_g,
        np.int32(Batch), np.int32(Channels), np.int32(Length), np.int32(kernel_size)
    )

    cl.enqueue_copy(queue, Y_host, y_g)

    # Эталон через PyTorch
    x_tensor = torch.from_numpy(X_host)
    w_tensor = torch.from_numpy(W_host).unsqueeze(1) # PyTorch ждет [Channels, 1, K] для depthwise
    b_tensor = torch.from_numpy(B_host)

    # В PyTorch делаем явный паддинг, наше OpenCL ядро делает его внутри "виртуально"
    x_padded = F.pad(x_tensor, (kernel_size - 1, 0))
    res_torch = F.conv1d(x_padded, w_tensor, bias=b_tensor, groups=Channels).numpy()

    # Проверка совпадения результатов
    np.testing.assert_allclose(Y_host, res_torch, rtol=1e-4, atol=1e-4)
    print("✅ Успешно! Каузальная свертка совпадает с PyTorch (без явного копирования паддинга!).")

    # =========================================================================
    # ТЕСТ 4: Универсальное транспонирование / перестановка осей (Permute)
    # Ситуация: [B, L, H, D] -> transpose(1, 2) -> [B, H, L, D]
    # =========================================================================
    print("\n[Тест 4] Проверка Универсального Транспонирования (Permute)...")
    B, L, H, D = 2, 32, 8, 64  # Настоящие размерности мини-LLM (Batch, Length, Heads, Head_Dim)
    
    src_shape = (B, L, H, D)
    # Порядок осей, который мы хотим получить на выходе:
    # Исходная ось 0 (B) остается на 0
    # Исходная ось 2 (H) встает на позицию 1
    # Исходная ось 1 (L) встает на позицию 2
    # Исходная ось 3 (D) остается на 3
    permute_axes = [0, 2, 1, 3]
    dst_shape = (B, H, L, D)
    
    total_elements = int(np.prod(src_shape))

    # Генерируем случайный 4D-тензор
    X_host = np.random.randn(*src_shape).astype(np.float32)
    Y_host = np.zeros(dst_shape, dtype=np.float32)

    # Читаем и компилируем permute.cl
    permute_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'permute.cl')
    with open(permute_path, 'r', encoding='utf-8') as f:
        permute_source = f.read()
    program_permute = cl.Program(ctx, permute_source).build()

    # Подготавливаем метаданные в формате C-массивов int32
    src_dims_np = np.array(src_shape, dtype=np.int32)
    dst_dims_np = np.array(dst_shape, dtype=np.int32)
    axes_np = np.array(permute_axes, dtype=np.int32)

    # Буферы OpenCL
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X_host)
    y_g = cl.Buffer(ctx, mf.WRITE_ONLY, Y_host.nbytes)
    src_dims_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_dims_np)
    dst_dims_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dst_dims_np)
    axes_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=axes_np)

    # Запускаем линейную сетку потоков по количеству элементов выходного тензора
    global_size = (total_elements,)

    program_permute.tensor_permute(
        queue, global_size, None,
        x_g, y_g,
        src_dims_g, dst_dims_g, axes_g,
        np.int32(len(src_shape)), np.int32(total_elements)
    )

    cl.enqueue_copy(queue, Y_host, y_g)

    # Эталон через PyTorch
    x_tensor = torch.from_numpy(X_host)
    res_torch = x_tensor.permute(*permute_axes).numpy()

    # Сверяем структуры и данные
    np.testing.assert_allclose(Y_host, res_torch, rtol=1e-5, atol=1e-5)
    print("✅ Успешно! Универсальное ядро перестановки осей 4D-тензора работает корректно.")


if __name__ == "__main__":
    test_all_matrix_operations()
