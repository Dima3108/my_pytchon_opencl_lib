import numpy as np
import pyopencl as cl
from core.tensor import Tensor
from core.module import Module  # <-- 1. ОБЯЗАТЕЛЬНО ИМПОРТИРУЕМ НАШ MODULE

# 2. Наследуем класс Linear от Module
class Linear(Module):
    def __init__(self, in_features, out_channels, nnclglob):
        super().__init__(nnclglob.ctx,nnclglob.queue)  # <-- ЖИЗНЕННО ВАЖНО: инициализируем родительский Module
        self.nnclglob = nnclglob
        self.in_features = in_features
        self.out_channels = out_channels
        
        bound = 1.0 / np.sqrt(in_features)
        w_init = np.random.uniform(-bound, bound, size=(out_channels, in_features)).astype(np.float32)
        b_init = np.zeros((out_channels,), dtype=np.float32)
        
        self.weight = Tensor.from_numpy(w_init, nnclglob.ctx, nnclglob.queue)
        self.bias = Tensor.from_numpy(b_init, nnclglob.ctx, nnclglob.queue)
        
    def forward(self, x: Tensor) -> Tensor:
        # ... (Ваш предыдущий код метода) ...
        x_2d = x.reshape(-1, self.in_features)
        x_3d = x_2d.unsqueeze(0)
        
        # ИСПРАВЛЕНИЕ: Явно передаем порядок осей [1, 0] для транспонирования матрицы весов!
        w_3d = self.nnclglob.permute(self.weight, [1, 0]).unsqueeze(0)
        
        res_3d = self.nnclglob.bmm(x_3d, w_3d)
        
        out_shape = list(x.shape)
        out_shape[-1] = self.out_channels
        return res_3d.reshape(out_shape)



# 3. Наследуем класс Embedding от Module
class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim, nnclglob):
        super().__init__()  # <-- ЖИЗНЕННО ВАЖНО: инициализируем родительский Module
        self.nnclglob = nnclglob
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        w_init = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.02
        self.weight = Tensor.from_numpy(w_init, nnclglob.ctx, nnclglob.queue)
        
    def forward(self, x_tokens_np) -> Tensor:
        # ... (Ваш текущий рабочий код метода forward остаётся БЕЗ изменений) ...
        real_total_tokens = x_tokens_np.size
        x_tokens_np = x_tokens_np.astype(np.int32).flatten()
        mf = cl.mem_flags
        tokens_buf = cl.Buffer(self.nnclglob.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_tokens_np)
        
        self._last_tokens_buf = tokens_buf
        self._last_total_tokens = real_total_tokens
        
        return self.nnclglob.embedding(tokens_buf, self.weight, real_total_tokens)


