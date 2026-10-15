import os
import sys
import numpy as np
import pyopencl as cl

# Добавляем корень проекта в пути импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.tensor import Tensor

def test_pure_tensor_allocation():
    print("\n" + "="*60)
    print("🔍 ИЗОЛИРОВАННЫЙ ТЕСТ ВЫДЕЛЕНИЯ ПАМЯТИ КЛАССА TENSOR")
    print("="*60)

    # Инициализация OpenCL окружения
    platforms = cl.get_platforms()[0]
    devices = platforms.get_devices()[0]
    ctx = cl.Context([devices])
    queue = cl.CommandQueue(ctx)

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ А: Выделение через пустой конструктор __init__
    # -------------------------------------------------------------------------
    requested_shape_1 = [8, 16]
    t1 = Tensor(requested_shape_1, ctx, queue)
    
    print("[Тест А] Выделение пустого тензора Tensor([8, 16]):")
    print(f"  • Ожидаемая форма (shape):  {requested_shape_1}")
    print(f"  • Фактическая форма (shape): {t1.shape}")
    print(f"  • Ожидаемый size:           {8 * 16}")
    print(f"  • Фактический size:          {t1.size}")
    
    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ Б: Выделение через фабрику from_numpy
    # -------------------------------------------------------------------------
    np_array = np.random.randn(11, 17).astype(np.float32)
    t2 = Tensor.from_numpy(np_array, ctx, queue)
    
    print("\n[Тест Б] Выделение тензора через Tensor.from_numpy с формой (8, 16):")
    print(f"  • Ожидаемая форма (shape):  {list(np_array.shape)}")
    print(f"  • Фактическая форма (shape): {t2.shape}")
    print(f"  • Ожидаемый size:           {np_array.size}")
    print(f"  • Фактический size:          {t2.size}")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_pure_tensor_allocation()
