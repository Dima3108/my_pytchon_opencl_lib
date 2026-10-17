import os
import sys
import numpy as np
import pyopencl as cl
import torch

# Добавляем корневую директорию проекта в путь поиска модулей Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from core.module import Module
from core.layers import Linear
from core.optimizer import AdamWOptimizer
from compiler.nncl_compiler import LocalRunner
from compiler.nncl_glob import NnClGlob
from compiler.host_returner import HostReturner

# Симулируем легкий математический модуль nnclloc для синтаксиса лямбды
class DummyNnclLoc:
    def sigmoid(self, x): return x
    def tanh(self, x): return x
nnclloc = DummyNnclLoc()

# ============================================================================
# СБОРКА ТЕСТОВОЙ МОДЕЛИ
# ============================================================================
class TrainableMemoryBlock(Module):
    def __init__(self, d_model, ctx, queue):
        super().__init__()
        self.ctx = ctx
        self.queue = queue
        self.nnclglob = NnClGlob(ctx, queue)
        self.runner = LocalRunner(ctx, queue, is_training=True)
        
        # Объявляем два полносвязных слоя (проекции гейтов)
        self.gate_proj = Linear(d_model, d_model, self.nnclglob)
        self.input_proj = Linear(d_model, d_model, self.nnclglob)
        
    def forward(self, x, memory_state, DYNAMIC_SEED, DROPOUT_P):
            # 1. Глобальные линейные проекции: [1, 4, 8] -> [1, 4, 8]
            gate_vectors_3d = self.gate_proj(x)
            input_vectors_3d = self.input_proj(x)
            
            # Убираем фиктивную ось батча 1 через БЕСПЛАТНЫЙ виртуальный squeeze,
            # чтобы перевести векторы из 3D [1, 4, 8] в плоскую 2D геометрию,
            # которая в точности совпадает с геометрией memory_state!
            gate_vectors = gate_vectors_3d.squeeze(0)
            input_vectors = input_vectors_3d.squeeze(0)
            
            # Выделяем целевые буферы во VRAM строго правильной формы [4, 8]
            shape_2d = memory_state.shape # Форма [4, 8]
            
            gate_A = Tensor(shape_2d, self.ctx, self.queue)
            gate_B = Tensor(shape_2d, self.ctx, self.queue)
            output = Tensor(shape_2d, self.ctx, self.queue)
            
            gate_A.require_grad()
            gate_B.require_grad()
            output.require_grad()
            
            # 2. ТОТАЛЬНЫЙ МОНОЛИТНЫЙ JIT-ФЬЮЗИНГ (Вся математика гейтов за 1 проход по VRAM!)
            # Передаем обычную, красивую текстовую строку инструкций.
            # Наш LocalRunner отфильтрует только нужные операнды, автоматически вплавит константы
            # и скомпилирует идеальное Си-ядро.
            self.runner(f"""
                gate_A = nnclloc.sigmoid(gate_vectors)
                gate_B = nnclloc.tanh(input_vectors)
                memory_state = memory_state * gate_A
                memory_state = memory_state + gate_B
                output = nnclloc.dropout(memory_state, {DROPOUT_P}, tid, {DYNAMIC_SEED})
            """)
            
            # Извлекаем обновленные тензоры обратно
            return self.runner.get_tensor('output'), self.runner.get_tensor('memory_state')





def test_full_training_loop_step():
    print("\n" + "="*80)
    print("🚀 ЗАПУСК С КВОЗНОГО ИНТЕГРАЦИОННОГО ТЕСТА ОБУЧЕНИЯ МОДЕЛИ (VRAM PIPELINE)")
    print("="*80)

    # Инициализация единого OpenCL окружения
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()[0]
    ctx = cl.Context([devices])
    queue = cl.CommandQueue(ctx)
    print(f"Тренировка пайплайна на устройстве: {devices.name}")

    # Геометрия модели: Batch=1, Length=4, d_model=8
    B, L, D = 1, 4, 8
    
    np.random.seed(42)
    init_x = np.random.randn(B, L, D).astype(np.float32)
    init_memory = np.ones((B * L, D), dtype=np.float32) # Исходное состояние памяти

    # Параметры Dropout и Оптимизатора
    SEED = 77777
    P_DROP = 0.2
    LR = 0.01

    # -------------------------------------------------------------------------
    # КОНТУР НАШЕЙ OPENCL БИБЛИОТЕКИ
    # -------------------------------------------------------------------------
    x_cl = Tensor.from_numpy(init_x, ctx, queue)
    memory_cl = Tensor.from_numpy(init_memory, ctx, queue)
    memory_cl.require_grad()

    # Создаем модель и вытаскиваем копии весов для PyTorch до шага оптимизации
    model = TrainableMemoryBlock(d_model=D, ctx=ctx, queue=queue)
    w_gate_init = HostReturner.to_cpu(model.gate_proj.weight)
    w_in_init = HostReturner.to_cpu(model.input_proj.weight)

    # Инициализируем AdamW, автоматически собирая параметры через Module.parameters()
    optimizer = AdamWOptimizer(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)

    print("\nШаг 1: Запуск Forward прохода всей модели...")
    output_cl, final_memory_cl = model(x_cl, memory_cl, DYNAMIC_SEED=SEED, DROPOUT_P=P_DROP)

    print("Шаг 2: Имитация прихода градиента ошибки сверху (Loss Backward)...")
    # Пусть от функции потерь пришел единичный вектор градиента ошибки по выходу
    grad_output = np.ones((B * L, D), dtype=np.float32)
    cl.enqueue_copy(queue, output_cl.grad_buffer, grad_output)
    queue.finish()
    print(f"результат:{grad_output}")
    print("Шаг 3: Развертывание автоматического обратного прохода градиентов...")
    model.runner.run_backward()

    # Скачиваем вычисленные градиенты для проверки
    cl_grad_w_gate = HostReturner.to_cpu_grad(model.gate_proj.weight)

    print("Шаг 4: Обновление параметров во VRAM через AdamWOptimizer...")
    optimizer.step()

    # Скачиваем обновленные веса для сравнения
    cl_updated_w_gate = HostReturner.to_cpu(model.gate_proj.weight)

    # -------------------------------------------------------------------------
    # ЭТАЛОННЫЙ КОНТУР В PYTORCH
    # -------------------------------------------------------------------------
    print("\nВычисление параллельного эталона на PyTorch для сверки...")
    t_x = torch.tensor(init_x, requires_grad=True)
    t_memory = torch.tensor(init_memory, requires_grad=True)

    t_gate_proj = torch.nn.Linear(D, D, bias=False)
    t_input_proj = torch.nn.Linear(D, D, bias=False)

    with torch.no_grad():
        t_gate_proj.weight.copy_(torch.from_numpy(w_gate_init))
        t_input_proj.weight.copy_(torch.from_numpy(w_in_init))

    # Прямой ход PyTorch (с выпрямлением, как в нашей bmm-реализации)
    t_x_2d = t_x.reshape(-1, D)
    t_gate_vecs = t_gate_proj(t_x_2d)
    t_input_vecs = t_input_proj(t_x_2d)

    t_gate_A = torch.sigmoid(t_gate_vecs)
    t_gate_B = torch.tanh(t_input_vecs)
    
    t_mem_fused = t_memory * t_gate_A + t_gate_B
    
    # Имитируем наш GPU детерминированный Dropout через маску выходов
    cl_output_np = HostReturner.to_cpu(output_cl)
    mask = (cl_output_np != 0.0).astype(np.float32)
    t_output = t_mem_fused * torch.from_numpy(mask) / (1.0 - P_DROP)

    # Обратный ход PyTorch
    t_output.backward(torch.ones((B * L, D)))

    # Шаг эталонного оптимизатора AdamW
    t_optimizer = torch.optim.AdamW(
        [t_gate_proj.weight, t_input_proj.weight], lr=LR, betas=(0.9, 0.999), weight_decay=0.0
    )
    t_optimizer.step()

    # -------------------------------------------------------------------------
    # ФИНАЛЬНАЯ ВЕРИФИКАЦИЯ (ASSERT_ALLCLOSE)
    # -------------------------------------------------------------------------
    print("\nСверка результатов:")
    
    # 1. Проверяем градиенты до обновления
    np.testing.assert_allclose(cl_grad_w_gate, t_gate_proj.weight.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Шаг Градиентов: Градиенты тяжелых линейных слоев сошлись с PyTorch!")

    # 2. Проверяем обновленные веса во VRAM
    np.testing.assert_allclose(cl_updated_w_gate, t_gate_proj.weight.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Шаг Оптимизации: Обновленные веса AdamW во VRAM совпали с PyTorch!")

    print("\n🎉 ПОЛНОЦЕННЫЙ ЦИКЛ ОБУЧЕНИЯ ВЕРИФИЦИРОВАН И УСПЕШНО ПРОЙДЕН!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_full_training_loop_step()
