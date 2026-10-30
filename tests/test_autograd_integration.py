# tests/test_autograd_integration.py
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
from compiler.nncl_compiler import LocalRunner
from compiler.nncl_glob import NnClGlob
# Имитируем легкий математический модуль nnclloc для синтаксиса лямбды внутри runner
class DummyNnclLoc:
    def sigmoid(self, x): return x
    def tanh(self, x): return x
nnclloc = DummyNnclLoc()

class JitIntegrationModel(Module):
    """
    Тестовая модель, которая объединяет глобальный Linear слой 
    и последующий fused-блок LocalRunner через метод __call2__.
    """
    def __init__(self, d_in, d_out, ctx, queue):
        super().__init__(ctx, queue)
        # Инициализируем глобальные операции с привязкой к единому контексту автограда
        self.nnclglob = NnClGlob(ctx, queue, autograd_ctx=self.autograd_ctx)    
        # Инициализируем локальный раннер в режиме обучения и жестко передаем autograd_ctx
        self.runner = LocalRunner(ctx, queue, is_training=True, autograd_ctx=self.autograd_ctx)
        # Создаем полносвязный слой, веса которого должны получить градиент
        self.linear_layer = Linear(d_in, d_out, self.nnclglob)

    def forward(self, x, memory_state):
        # 1. Глобальная линейная проекция (записывает bmm_backward в autograd_ctx)
        proj_vectors_3d = self.linear_layer(x)
        
        # 2. Виртуальный squeeze для перевода 3D -> 2D (геометрия memory_state)
        gate_vectors = proj_vectors_3d.squeeze(0)
        gate_vectors.autograd_ctx = self.autograd_ctx
        # Выделяем целевой буфер под выходной тензор JIT-блока
        output = Tensor(memory_state.shape, self.ctx, self.queue)
        output.require_grad()
        output.autograd_ctx = self.autograd_ctx
        #output.require_grad()
        gate_A=Tensor(memory_state.shape,self.ctx,self.queue)
        # 3. ТОТАЛЬНЫЙ МОНОЛИТНЫЙ JIT-ФЬЮЗИНГ ЧЕРЕЗ __call2__
        # Вызов скомпилирует раздельные ядра, выполнит forward и зарегистрирует backward callback
        self.runner.__call2__(f"""
            gate_A = nnclloc.sigmoid(gate_vectors)
            memory_state = memory_state * gate_A
            output = memory_state + gate_A
        """)

        return self.runner.get_tensor('output')


def test_skvoznoi_autograd_pipeline():
    print("\n" + "="*80)
    print("🚀 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА: ПЕРЕДАЧА ГРАДИЕНТОВ ИЗ LOCALRUNNER В LINEAR")
    print("="*80)

    # 1. Инициализация OpenCL окружения
    # 1. Инициализация OpenCL окружения
    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # === ДОБАВЬТЕ ЭТУ СТРОКУ СЮДА ДЛЯ РЕГИСТРАЦИИ ОЧЕРЕДИ ===
    from core.queue_registry import DeviceQueueRegistry
    DeviceQueueRegistry.set_current(queue)
    
    print(f"Тестирование на устройстве: {device.name}")


    # Размеры: Batch=1, Length=4, d_model=8
    B, L, D = 1, 4, 8
    
    np.random.seed(42)
    init_x = np.random.randn(B, L, D).astype(np.float32)
    init_memory = np.ones((B * L, D), dtype=np.float32)

    # 2. КОНТУР НАШЕЙ OPENCL БИБЛИОТЕКИ
    x_cl = Tensor.from_numpy(init_x, ctx, queue)
    memory_cl = Tensor.from_numpy(init_memory, ctx, queue)
    memory_cl.require_grad()

    # Создаем модель и вытаскиваем начальные веса для PyTorch
    model = JitIntegrationModel(d_in=D, d_out=D, ctx=ctx, queue=queue)
    model.train()
    w_linear_init = model.linear_layer.weight.to_numpy()
    model.linear_layer.weight.require_grad() 
    # Шаг 1: Прямой ход нашей модели
    output_cl = model(x_cl, memory_cl)
    from core.queue_registry import DeviceQueueRegistry
    active_queue = DeviceQueueRegistry.get_current()
    # Шаг 2: Имитация прихода градиента ошибки (Loss Backward)
    grad_output = np.ones((B * L, D), dtype=np.float32)
    cl.enqueue_copy(active_queue, output_cl.grad_buffer, grad_output)
    #queue.finish()
    active_queue.finish()
    # Шаг 3: Развертывание ЕДИНОЙ очереди команд автограда
    # 1. Вызовется runner_step_backward_callback() (вычислит градиенты в раннере)
    # 2. Вызовется bmm_backward слоев выше (вычислит дельты весов Linear)
    model.autograd_ctx.run_backward()

    # Скачиваем полученный градиент весов из VRAM
    from compiler.host_returner import HostReturner
    cl_grad_w_linear = HostReturner.to_cpu_grad(model.linear_layer.weight)
    #cl_grad_w_linear=model.linear_layer.weight
    print(f"'{cl_grad_w_linear[:10]}'\n")


    # 3. ЭТАЛОННЫЙ КОНТУР В PYTORCH
    print("\nВычисление параллельного эталона на PyTorch для сверки...")
    t_x = torch.tensor(init_x, requires_grad=True)
    t_memory = torch.tensor(init_memory, requires_grad=True)
    
    t_linear = torch.nn.Linear(D, D, bias=False)
    with torch.no_grad():
        t_linear.weight.copy_(torch.from_numpy(w_linear_init))

    # Прямой ход PyTorch
    t_x_2d = t_x.reshape(-1, D)
    t_proj = t_linear(t_x_2d)
    
    t_gate_A = torch.sigmoid(t_proj)
    t_fused_mem = t_memory * t_gate_A
    t_output = t_fused_mem + t_gate_A

    # Обратный ход PyTorch
    t_output.backward(torch.ones((B * L, D)))
    pytorch_grad_w_linear = t_linear.weight.grad.numpy()


    # 4. ВЕРИФИКАЦИЯ (ASSERT)
    print("\nСверка результатов градиентов весов:")
    try:
        np.testing.assert_allclose(cl_grad_w_linear, pytorch_grad_w_linear, rtol=1e-4, atol=1e-4)
        print("✅ УСПЕХ: Градиенты весов Linear слоя полностью совпали с PyTorch!")
        print("🔗 Связка AutogradContext -> LocalRunner (__call2__) -> NnClGlob работает корректно.")
    except AssertionError as e:
        print("❌ ОШИБКА: Обнаружено расхождение в градиентах весов!")
        raise e

if __name__ == "__main__":
    test_skvoznoi_autograd_pipeline()

