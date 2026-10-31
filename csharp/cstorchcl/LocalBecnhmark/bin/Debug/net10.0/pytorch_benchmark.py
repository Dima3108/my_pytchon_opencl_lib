import time
import torch

def print_system_info():
    print("=== ЗАПУСК ИСПРАВЛЕННОГО БЕНЧМАРКА: PYTORCH CUDA (WINDOWS FUSION) ===")
    print(f"Версия PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"Выбран девайс: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA не доступна! Проверьте установку PyTorch с поддержкой GPU.")

# Базовая функция (Раздельные ядра)
def math_pipeline(a, b, c, q):
    a = torch.tanh(a)
    c = torch.cos(b)
    a = a / q
    c = torch.nn.functional.dropout(c, p=0.2, training=True)
    b = b + 0.01
    return a, b, c

# Сплавленная функция через TorchScript JIT Compiler (Идеально для Windows!)
@torch.jit.script
def compiled_pipeline(a, b, c, q: float):
    a = torch.tanh(a)
    c = torch.cos(b)
    a = a / q
    c = torch.nn.functional.dropout(c, p=0.2, training=True)
    b = b + 0.01
    return a, b, c

def main():
    print_system_info()
    
    size_a, size_b, size_c = 100_000, 80_000, 120_000
    iterations = 1000
    q = 1.5
    
    torch.manual_seed(42)
    device = "cuda"
    
    tensor_a = torch.ones(size_a, dtype=torch.float32, device=device)
    tensor_b = torch.full((size_b,), 0.5, dtype=torch.float32, device=device)
    tensor_c = torch.full((size_c,), -1.0, dtype=torch.float32, device=device)
    
    a1, b1, c1 = tensor_a.clone(), tensor_b.clone(), tensor_c.clone()
    a2, b2, c2 = tensor_a.clone(), tensor_b.clone(), tensor_c.clone()

    # =========================================================================
    # ТЕСТ 1: СТАНДАРТНЫЙ РЕЖИМ (Eager Mode — Раздельные ядра)
    # =========================================================================
    print(f"\nТест 1: Запуск {iterations} итераций (Eager Mode / Unfused)...")
    
    # Разогрев
    for _ in range(10):
        _ = math_pipeline(a1, b1, c1, q)
    torch.cuda.synchronize()
    
    start_unfused = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            a1, b1, c1 = math_pipeline(a1, b1, c1, q)
            
    torch.cuda.synchronize()
    end_unfused = time.perf_counter()
    total_unfused_ms = (end_unfused - start_unfused) * 1000

    # =========================================================================
    # ТЕСТ 2: КРОСС-ПЛАТФОРМЕННЫЙ JIT-РЕЖИМ (TorchScript Fusion)
    # =========================================================================
    print(f"Тест 2: Запуск {iterations} итераций (TorchScript JIT / Fused)...")
    
    # Разогрев и компиляция TorchScript ядра на лету
    _ = compiled_pipeline(a2, b2, c2, q)
    torch.cuda.synchronize()
    
    start_fused = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            a2, b2, c2 = compiled_pipeline(a2, b2, c2, q)
            
    torch.cuda.synchronize()
    end_fused = time.perf_counter()
    total_fused_ms = (end_fused - start_fused) * 1000

    # =========================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ БЕНЧМАРКА
    # =========================================================================
    avg_unfused_ms = total_unfused_ms / iterations
    avg_fused_ms = total_fused_ms / iterations
    speedup = total_unfused_ms / total_fused_ms
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ (PYTORCH ON WINDOWS)")
    print("="*60)
    print(f"Раздельные ядра (Eager Mode):")
    print(f"  -> Общее время:          {total_unfused_ms:.2f} мс")
    print(f"  -> Среднее на итерацию:  {avg_unfused_ms:.4f} мс")
    
    print(f"\nСплавленное ядро (TorchScript JIT):")
    print(f"  -> Общее время:          {total_fused_ms:.2f} мс")
    print(f"  -> Среднее на итерацию:  {avg_fused_ms:.4f} мс")
    
    print("-"*60)
    print(f"Ускорение за счет PyTorch JIT-Fusion: {speedup:.2f}x раз(а)!")
    print("="*60)

if __name__ == "__main__":
    main()

