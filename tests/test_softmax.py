import os
import numpy as np
import pyopencl as cl
import torch

def test_softmax_forward():
    # 1. Задаем геометрию тензора [B * H * L, SEQ_LEN]
    # Настраиваем BLOCK_SIZE под размер l_mem[256] в вашем ядре
    BATCH_ROWS = 32   # Количество строк (B * H * L)
    SEQ_LEN = 512     # Длина последовательности (количество элементов в строке)
    BLOCK_SIZE = 256  # Количество потоков в рабочей группе (должно быть <= l_mem)

    print(f"Запуск теста Softmax. Размерность: [{BATCH_ROWS}, {SEQ_LEN}]")

    # 2. Генерируем случайные входные данные в диапазоне [0; 1]
    x_host = np.random.rand(BATCH_ROWS, SEQ_LEN).astype(np.float32)
    y_host = np.zeros_like(x_host)

    # 3. Инициализируем OpenCL контекст и очередь команд
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    print(f"Используемое устройство: {device.name}")

    # 4. Читаем код ядра из папки cl_kernels
    kernel_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'Softmax.cl')
    with open(kernel_path, 'r', encoding='utf-8') as f:
        kernel_source = f.read()

    # 5. Компилируем программу OpenCL
    program = cl.Program(ctx, kernel_source).build()

    # 6. Создаем буферы в видеопамяти (VRAM)
    mf = cl.mem_flags
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_host)
    y_g = cl.Buffer(ctx, mf.WRITE_ONLY, y_host.nbytes)

    # 7. Вызываем OpenCL ядро
    # Глобальный размер (Global Size): выставляем потоков столько, чтобы каждая группа из 256 
    # потоков обрабатывала ровно одну строку матрицы.
    global_size = (BATCH_ROWS * BLOCK_SIZE,)
    local_size = (BLOCK_SIZE,)

    program.softmax_forward(
        queue, 
        global_size, 
        local_size, 
        x_g, 
        y_g, 
        np.int32(SEQ_LEN)
    )

    # 8. Читаем результат обратно в оперативную память (Host)
    cl.enqueue_copy(queue, y_host, y_g)

    # 9. Считаем эталонное значение через оригинальный PyTorch Softmax
    x_tensor = torch.from_numpy(x_host)
    y_torch = torch.softmax(x_tensor, dim=-1).numpy()

    # 10. Сверяем результаты на близость (учитывая погрешность float32)
    try:
        np.testing.assert_allclose(y_host, y_torch, rtol=1e-5, atol=1e-5)
        print("✅ ТЕСТ ПРОЙДЕН! Результаты вашего OpenCL ядра полностью совпадают с PyTorch.")
    except AssertionError as e:
        print("❌ ТЕСТ ПРОВАЛЕН! Обнаружены расхождения в вычислениях:")
        print(e)
        
        # Выведем для наглядности первую строку
        print("\nПервая строка (Ваш OpenCL):", y_host[0][:5])
        print("Первая строка (Оригинальный PyTorch):", y_torch[0][:5])

if __name__ == "__main__":
    test_softmax_forward()
