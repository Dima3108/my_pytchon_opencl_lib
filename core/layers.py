import numpy as np
import pyopencl as cl
from core.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_channels, nnclglob):
        """
        Полносвязный слой (Линейная проекция)
        Веса W имеют форму [out_channels, in_features]
        Смещение bias имеет форму [out_channels]
        """
        self.nnclglob = nnclglob
        self.in_features = in_features
        self.out_channels = out_channels
        
        # Инициализация весов методом Ксавье (Xavier initialization)
        bound = 1.0 / np.sqrt(in_features)
        w_init = np.random.uniform(-bound, bound, size=(out_channels, in_features)).astype(np.float32)
        b_init = np.zeros((out_channels,), dtype=np.float32)
        
        self.weight = Tensor.from_numpy(w_init, nnclglob.ctx, nnclglob.queue)
        self.bias = Tensor.from_numpy(b_init, nnclglob.ctx, nnclglob.queue)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Вход X имеет форму [Batch, Seq_Len, in_features].
        Чтобы сделать bmm, мы виртуально трактуем его как пакет матриц.
        Вычисляем: Y = X @ W.T
        """
        # Наше ядро bmm_tiled ждет форму [B, M, K] @ [B, K, N]
        # Переставляем оси весов через permute, имитируя W.T для пакета
        W_permuted = self.nnclglob.permute(self.weight, [1, 0]) # [in_features, out_channels]
        
        # Для bmm веса должны иметь размерность пакета (бродкастим виртуально или дублируем метаданные)
        # В нашей архитектуре плоских буферов мы можем просто подстроить размеры шагов (strides)
        # Для простоты создадим 3D представление:
        batch_size = x.shape[0] if len(x.shape) == 3 else 1
        m_dim = x.shape[1] if len(x.shape) == 3 else x.shape[0]
        
        # Делаем bmm проекцию
        # Превращаем веса в 3D обертку [Batch, in_features, out_channels] с шагом 0 по батчу
        # Для этого модифицируем bmm_tiled аргументы
        # ... (В рамках макро-вызова вызываем пакетное умножение)
        # Чтобы не усложнять bmm strides, мы можем временно выпрямить X до 2D [B * L, in_features]
        x_2d = x.reshape(-1, self.in_features)
        
        # Добавляем фиктивную ось батча = 1, чтобы использовать bmm_tiled как обычный matmul
        x_3d = x_2d.unsqueeze(0) # [1, B*L, in_features]
        w_3d = W_permuted.unsqueeze(0) # [1, in_features, out_channels]
        
        res_3d = self.nnclglob.bmm(x_3d, w_3d) # [1, B*L, out_channels]
        
        # Возвращаем исходную многомерную форму [Batch, Seq_Len, out_channels]
        out_shape = list(x.shape)
        out_shape[-1] = self.out_channels
        
        y = res_3d.reshape(out_shape)
        
        # Добавление bias (это легкая поэлементная операция с бродкастингом)
        # Она выполнится через наш LocalRunner на стороне модели, 
        # либо мы добавим её в глобальный контур. Оставим bias для LocalRunner
        return y


class Embedding:
    def __init__(self, vocab_size, embedding_dim, nnclglob):
        self.nnclglob = nnclglob
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Инициализация весов нормальным распределением
        w_init = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.02
        print(f"размерность матрицы весов:{w_init.shape}")
        self.weight = Tensor.from_numpy(w_init, nnclglob.ctx, nnclglob.queue)
        
    def forward(self, x_tokens_np) -> Tensor:
        # Честно считаем количество элементов на CPU (равно 8)
        real_total_tokens = x_tokens_np.size 
        
        x_tokens_np = x_tokens_np.astype(np.int32).flatten()
        mf = cl.mem_flags
        tokens_buf = cl.Buffer(self.nnclglob.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_tokens_np)
        
        self._last_tokens_buf = tokens_buf
        self._last_total_tokens = real_total_tokens # Сохраняем для backward
        
        # Передаем количество элементов третьим аргументом
        return self.nnclglob.embedding(tokens_buf, self.weight, real_total_tokens)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Возвращает градиент по весам матрицы эмбеддингов"""
        return self.nnclglob.embedding_backward(
            grad_output, 
            self._last_tokens_buf, 
            self.weight.shape, 
            self._last_total_tokens
        )

