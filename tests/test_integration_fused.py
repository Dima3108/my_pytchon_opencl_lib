import os
import sys
import numpy as np
import pyopencl as cl
import torch

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from compiler.nncl_compiler import LocalRunner
from compiler.host_returner import HostReturner

# Симулируем модуль nnclloc для синтаксиса лямбды внутри теста
class DummyNnclLoc:
    def sigmoid(self, x): return x
    def tanh(self, x): return x
nnclloc = DummyNnclLoc()

def test_integration_fused_pipeline():
    print("\n" + "="*70)
    print("🚀 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА: ТЕНЗОР -> LOCALRUNNER -> HOSTRETURNER")
    print("="*70)

    # 1. Инициализация OpenCL окружения
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("OpenCL платформы не найдены.")
    devices = platforms[0].get_devices()
    if not devices:
        raise RuntimeError("OpenCL устройства не найдены.")
    
    ctx = cl.Context([devices[0]])
    queue = cl.CommandQueue(ctx)
    print(f"Используемое устройство девайса: {devices[0].name}")

    # 2. Задаем геометрию тестовых данных (плоский срез d_model=8)
    shape = (8,)
    
    # Исходные случайные массивы на CPU
    init_gate_A = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 1.5], dtype=np.float32)
    init_memory = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # 3. Создаем "чистые" объекты Tensor в нашей OpenCL библиотеке
    gate_A = Tensor.from_numpy(init_gate_A, ctx, queue)
    memory_state = Tensor.from_numpy(init_memory, ctx, queue)

    # Включаем режим обучения (выделение буферов под градиенты)
    gate_A.require_grad()
    memory_state.require_grad()

    # Инициализируем компилятор мета-программы в режиме обучения
    runner = LocalRunner(ctx, queue, is_training=True)

    # 4. ВЫПОЛНЕНИЕ ПРЯМОГО ПРОХОДА (Forward JIT-Fusion)
    print("\nВыполнение прямого прохода на GPU...")
    runner(lambda: gate_A == nnclloc.sigmoid(gate_A))
    runner(lambda: memory_state == memory_state * gate_A)

    # 5. ИМИТАЦИЯ ОБРАТНОГО ПРОХОДА (Backward)
    # ЯВНО ОБЪЯВЛЯЕМ вектор пришедшего градиента (чистые единицы)
    init_grad_output = np.ones(shape, dtype=np.float32)

    # Загружаем единичный градиент в memory_state.grad_buffer
    cl.enqueue_copy(queue, memory_state.grad_buffer, init_grad_output)
    
    # ЖИЗНЕННО ВАЖНО: Очищаем (обнуляем) буфер градиентов для gate_A перед бэквардом,
    # чтобы операция += в JIT-ядре не прибавляла мусор к накопленным значениям
    cl.enqueue_fill_buffer(queue, gate_A.grad_buffer, np.float32(0.0), 0, gate_A.nbytes)
    
    queue.finish()

    print("Выполнение автоматического обратного прохода градиентов...")
    runner.run_backward()

    # 6. ВЫКАЧИВАНИЕ И ОТЛАДКА ДАННЫХ ИСКЛЮЧИТЕЛЬНО ЧЕРЕЗ HostReturner
    print("\nСчитывание и визуализация результатов через внешний HostReturner:")
    HostReturner.print_tensor(memory_state, name="Финальное состояние памяти (Memory State)", print_grad=True)
    HostReturner.print_tensor(gate_A, name="Гейт А после Sigmoid", print_grad=True)

    # 7. МАТЕМАТИЧЕСКАЯ ВЕРИФИКАЦИЯ ЧЕРЕЗ ЭТАЛОННЫЙ PYTORCH
    print("Верификация расчетов с оригинальным PyTorch...")
    t_gate_A = torch.tensor(init_gate_A, requires_grad=True)
    t_memory_state = torch.tensor(init_memory, requires_grad=True)

    # Точный повтор цепочки на PyTorch
    t_gate_A_fused = torch.sigmoid(t_gate_A)
    t_memory_state_fused = t_memory_state * t_gate_A_fused
    
    # Запуск эталонного backward
    t_memory_state_fused.backward(torch.ones(shape))

    # Извлекаем массивы из OpenCL при помощи HostReturner для проверок
    cl_res_memory = HostReturner.to_cpu(memory_state)
    cl_grad_gate_A = HostReturner.to_cpu_grad(gate_A)
    print("\n\n")
    HostReturner.print_tensor(tensor=gate_A, print_grad=True)
    # Сверяем результаты прямого и обратного хода
    np.testing.assert_allclose(cl_res_memory, t_memory_state_fused.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cl_grad_gate_A, t_gate_A.grad.numpy(), rtol=1e-5, atol=1e-5)
    
    print("\n✅ ИНТЕГРАЦИОННЫЙ ТЕСТ УСПЕШНО ПРОЙДЕН!")
    print("Связка Tensor -> LocalRunner -> HostReturner работает математически верно.")
    print("="*70 + "\n")



if __name__ == "__main__":
    test_integration_fused_pipeline()
    #test_softmax_glob_pipeline()
