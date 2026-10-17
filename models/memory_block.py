from core.module import Module
from core.layers import Linear
import compiler.nncl_compiler as compiler

class LinearMemoryBlock(Module):
    def __init__(self, d_model, ctx, queue):
        super().__init__()
        self.ctx = ctx
        self.queue = queue
        
        # Инициализируем глобальную очередь команд (тяжелые операции)
        self.nnclglob = compiler.NnClGlob(ctx, queue)
        # Инициализируем JIT-компилятор локальных операций
        self.runner = compiler.LocalRunner(ctx, queue, is_training=True)
        
        # Декларативно объявляем тяжелые слои проекций гейтов
        self.gate_proj = Linear(d_model, d_model, self.nnclglob)
        self.input_proj = Linear(d_model, d_model, self.nnclglob)
        
    def forward(self, x, memory_state, DYNAMIC_SEED, DROPOUT_P):
        # 1. Глобальные линейные проекции: [1, 4, 8] -> [1, 4, 8]
        gate_vectors = self.gate_proj(x)
        input_vectors = self.input_proj(x)
        
        # Аллоцируем буферы под результаты активаций гейтов
        gate_A = Tensor(gate_vectors.shape, self.ctx, self.queue)
        gate_B = Tensor(input_vectors.shape, self.ctx, self.queue)
        gate_A.require_grad()
        gate_B.require_grad()
        
        # ---------------------------------------------------------------------
        # КОНТУР JIT 1: Локальное вычисление активаций (У них одинаковый 3D размер)
        # ---------------------------------------------------------------------
        self.runner("""
            gate_A = nnclloc.sigmoid(gate_vectors)
            gate_B = nnclloc.tanh(input_vectors)
        """)
        
        # 2. ГЛОБАЛЬНАЯ ОПЕРАЦИЯ: Пакетное матричное умножение (BMM)
        # Чтобы перемножить memory_state [4, 8] и gate_A, мы превращаем
        # memory_state в 3D обертку [1, 4, 8] через unsqueeze(0)
        memory_3d = memory_state.unsqueeze(0)
        
        # Для честной проекции BMM нам нужно транспонировать последние оси у gate_A
        # [1, 4, 8] @ [1, 8, 4] ->. Чтобы получить чистый [1, 4, 8] поэлементно,
        # если это была задумка взвешивания, мы выполняем bmm.
        # Если цель — взвесить строки, вызываем bmm:
        gate_A_T = self.nnclglob.permute(gate_A, [0, 2, 1])
        fused_memory_3d = self.nnclglob.bmm(memory_3d, gate_A_T)
        
        # Возвращаем к исходной 2D форме
        fused_memory = fused_memory_3d.reshape(memory_state.shape)
        
        # Аллоцируем буфер под итоговый выход
        output = Tensor(memory_state.shape, self.ctx, self.queue)
        output.require_grad()
        
        # ---------------------------------------------------------------------
        # КОНТУР JIT 2: Локальное сложение результатов и Dropout (Размеры совпадают)
        # ---------------------------------------------------------------------
        # Превращаем gate_B из 3D [1, 4, 8] в 2D [4, 8] через reshape
        gate_B_2d = gate_B.reshape(memory_state.shape)
        
        self.runner(f"""
            memory_state = fused_memory + gate_B_2d
            output = nnclloc.dropout(memory_state, {DROPOUT_P}, tid, {DYNAMIC_SEED})
        """)
        
        return self.runner.get_tensor('output'), self.runner.get_tensor('memory_state')
