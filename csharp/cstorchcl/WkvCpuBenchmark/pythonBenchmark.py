import time
import torch

# Убедитесь, что используется CPU
device = torch.device("cpu")
device=torch.device("cuda")
@torch.jit.script
def wkv_forward_pytorch(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, T, C = k.shape
    y = torch.empty_like(k)
    
    aa = torch.zeros((B, C), dtype=k.dtype, device=k.device)
    bb = torch.zeros((B, C), dtype=k.dtype, device=k.device)
    pp = torch.full((B, C), -1e38, dtype=k.dtype, device=k.device)
    
    for t in range(T):
        kk = k[:, t, :]
        vv = v[:, t, :]
        
        # 1. Выход y_t с учетом бонуса u
        ww = u + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        y[:, t, :] = (e1 * aa + e2 * vv) / (e1 * bb + e2)
        
        # 2. Обновление рекуррентного состояния с затуханием w
        ww = w + pp
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p
        
    return y

def main():
    B, T, C = 4, 512, 1024
    iterations = 100
    
    print(f"PyTorch version: {torch.__version__} | Threads: {torch.get_num_threads()}")
    print(f"Shape: B={B}, T={T}, C={C} | Iterations: {iterations}\n")
    
    torch.manual_seed(42)
    w = -torch.exp(torch.rand(C, device=device) * 2.0 - 1.0)
    u = torch.rand(C, device=device) * 2.0 - 1.0
    k = torch.rand((B, T, C), device=device) * 2.0 - 1.0
    v = torch.rand((B, T, C), device=device) * 2.0 - 1.0
    
    with torch.inference_mode():
        # Прогрев JIT
        for _ in range(10):
            y = wkv_forward_pytorch(w, u, k, v)
            
        start = time.perf_counter()
        for _ in range(iterations):
            y = wkv_forward_pytorch(w, u, k, v)
        end = time.perf_counter()
        
    total_ms = (end - start) * 1000.0
    it_per_sec = iterations / (end - start)
    print(f"[PyTorch JIT CPU] Total time: {total_ms:.2f} ms")
    print(f"[PyTorch JIT CPU] Speed:      {it_per_sec:.2f} it/s")
    print(f"Check y[0,0,0]: {y[0,0,0].item():.6f}, y[-1,-1,-1]: {y[-1,-1,-1].item():.6f}")

if __name__ == "__main__":
    main()
