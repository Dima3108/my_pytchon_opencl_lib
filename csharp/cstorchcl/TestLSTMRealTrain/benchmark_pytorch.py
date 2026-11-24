import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_xavier_cstorchcl(weight: torch.Tensor, in_features: int, out_features: int):
    bound = math.sqrt(6.0) / math.sqrt(in_features + out_features)
    with torch.no_grad():
        weight.uniform_(-bound, bound)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: [Batch, NumHeads, SeqLen, HeadDim]
        seq_len = q.shape[2]
        positions = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # [SeqLen, HeadDim // 2]
        cos = freqs.cos()[None, None, :, :]            # [1, 1, SeqLen, HeadDim // 2]
        sin = freqs.sin()[None, None, :, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            # Разбиваем на четные (2i) и нечетные (2i+1) координаты, как в вашем OpenCL-ядре
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            out_even = x_even * cos - x_odd * sin
            out_odd = x_even * sin + x_odd * cos
            return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, is_causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)
        self.rope = RotaryEmbedding(self.head_dim)

        for proj in (self.wq, self.wk, self.wv, self.wo):
            init_xavier_cstorchcl(proj.weight, d_model, d_model)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Применяем RoPE к Q и K
        q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, d_model, bias=True)

        init_xavier_cstorchcl(self.w_gate_up.weight, d_model, intermediate_dim)
        init_xavier_cstorchcl(self.w_down.weight, intermediate_dim, d_model)
        nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, intermediate_dim: int, is_causal: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, num_heads, is_causal)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x))
        return h + self.ffn(self.norm2(h))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== БЕНЧМАРК PYTORCH С ROPE (XAVIER INIT) | Девайс: {torch.cuda.get_device_name(0)} ===")

    torch.manual_seed(42)
    batch, seq_len, d_model, num_heads, intermediate_dim = 4, 64, 128, 4, 344

    x = (torch.rand(batch, seq_len, d_model, device=device) * 2.0 - 1.0) * 0.5
    x.requires_grad_(True)
    target = torch.sin(x.detach()) + 0.5 * x.detach()

    block = TransformerBlock(d_model, num_heads, intermediate_dim, is_causal=True).to(device)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01)

    epochs = 100
    total_fwd, total_bwd, total_step, count = 0.0, 0.0, 0.0, 0

    for epoch in range(epochs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        t_fwd0 = time.perf_counter()
        pred = block(x)
        torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()

        loss = F.mse_loss(pred, target)
        loss_val = loss.item()

        t_bwd0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd1 = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        fwd_ms = (t_fwd1 - t_fwd0) * 1000.0
        bwd_ms = (t_bwd1 - t_bwd0) * 1000.0
        step_ms = (t1 - t0) * 1000.0

        if epoch >= 5:
            total_fwd += fwd_ms
            total_bwd += bwd_ms
            total_step += step_ms
            count += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Эпоха {epoch + 1:02d}/{epochs} | Loss: {loss_val:.6f} | Fwd: {fwd_ms:.2f} мс | Bwd: {bwd_ms:.2f} мс | Шаг: {step_ms:.2f} мс")

    avg_step = total_step / count
    print("\n================================================================")
    print(f"[PyTorch + RoPE] Среднее Fwd: {total_fwd / count:.3f} мс | Bwd: {total_bwd / count:.3f} мс | Шаг: {avg_step:.3f} мс")
    print(f"[PyTorch + RoPE] Пропускная способность: {(batch * seq_len) / (avg_step / 1000.0):.0f} токенов/сек")
    print("================================================================")


if __name__ == "__main__":
    main()
