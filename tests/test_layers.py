import os
import sys
import numpy as np
import pyopencl as cl
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tensor import Tensor
from compiler.nncl_glob import NnClGlob
from core.layers import Linear, Embedding
from compiler.host_returner import HostReturner

def test_global_layers_pipeline():
    print("\n" + "="*70)
    print("🧪 ЗАПУСК КОМПЛЕКСНОГО ТЕСТА ГЛОБАЛЬНЫХ СЛОЕВ: EMBEDDING & LINEAR")
    print("="*70)

    # Инициализация OpenCL
    platforms = cl.get_platforms()[0]
    devices = platforms.get_devices()[0]
    ctx = cl.Context([devices])
    queue = cl.CommandQueue(ctx)
    nnclglob = NnClGlob(ctx, queue)
    print(f"Тестирование слоев на устройстве: {devices.name}")

    # =========================================================================
    # 1. ТЕСТИРОВАНИЕ СЛОЯ EMBEDDING (С ДЕTAЛЬНЫМ ЛОГИРОВАНИЕМ)
    # =========================================================================
    print("\n[Проверка слоя Embedding]...")
    VOCAB_SIZE, EMBED_DIM = 50, 16
    BATCH, SEQ_LEN = 2, 4
    
    # Случайные ID токенов
    np.random.seed(42)
    tokens_np = np.random.randint(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN))
    print(f"🔍 ЛОГ CPU: Входные токены tokens_np.shape = {tokens_np.shape}, total_size = {tokens_np.size}")
    
    emb_layer = Embedding(VOCAB_SIZE, EMBED_DIM, nnclglob)
    print(f"🔍 ЛОГ CPU: Матрица весов emb_layer.weight.shape = {emb_layer.weight.shape}")
    
    # Вызов прямого прохода нашей OpenCL-библиотеки
    # === Внутри tests/test_layers.py ===
    y_emb = emb_layer.forward(tokens_np) # Теперь вернет форму [8, 16] строго!
    print(f"🔍 ЛОГ CPU: Матрица y_emb.shape = {y_emb.shape}({y_emb.size})")
    # Имитируем градиент ошибки (создастся размером строго 8 * 16 = 128 элементов)
    grad_out_emb = np.random.randn(*y_emb.shape).astype(np.float32)
    grad_out_emb_cl = Tensor.from_numpy(grad_out_emb, ctx, queue)
    
    grad_weight_emb = emb_layer.backward(grad_out_emb_cl)
    
    cl_res_emb = HostReturner.to_cpu(y_emb)
    cl_grad_w_emb = HostReturner.to_cpu(grad_weight_emb)

    # Эталон в PyTorch
    t_emb = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    with torch.no_grad():
        t_emb.weight.copy_(torch.from_numpy(HostReturner.to_cpu(emb_layer.weight)))
        
    t_tokens = torch.from_numpy(tokens_np).long()
    t_y_emb = t_emb(t_tokens) # Форма PyTorch: [2, 4, 16]
    
    # ПРАВИЛЬНЫЙ РЕШЕЙП ДЛЯ PYTORCH: Переводим наши 128 элементов в 3D куб
    print(f"{BATCH*SEQ_LEN},{SEQ_LEN*EMBED_DIM},{EMBED_DIM*BATCH},{EMBED_DIM*SEQ_LEN*BATCH}")
    grad_out_emb_torch = grad_out_emb.reshape(BATCH, SEQ_LEN, EMBED_DIM)
    t_y_emb.backward(torch.from_numpy(grad_out_emb_torch))

    
    # === ВНУТРИ ФАЙЛА tests/test_layers.py (Строка 69) ===
    # Выпрямляем эталон PyTorch из [2, 4, 16] в плоские [8, 16] для честной сверки с OpenCL
    np.testing.assert_allclose(
        cl_res_emb, 
        t_y_emb.detach().numpy().reshape(-1, EMBED_DIM), 
        rtol=1e-4, 
        atol=1e-4
    )
    print("  ✅ Embedding Forward: Данные выборки токенов полностью сошлись!")

    print("  ✅ Embedding Forward: Данные выборки токенов полностью сошлись!")
    
    np.testing.assert_allclose(cl_grad_w_emb, t_emb.weight.grad.numpy(), rtol=1e-4, atol=1e-4)
    print("  ✅ Embedding Backward: Градиенты матрицы весов сошлись с PyTorch!")


    # =========================================================================
    # 2. ТЕСТИРОВАНИЕ СЛОЯ LINEAR
    # =========================================================================
    print("\n[Проверка слоя Linear]...")
    IN_FEAT, OUT_CH = 16, 8
    
    lin_layer = Linear(IN_FEAT, OUT_CH, nnclglob)
    # Подаем выход эмбеддинга на вход линейному слою
    y_linear = lin_layer.forward(y_emb)
    
    cl_res_lin = HostReturner.to_cpu(y_linear)
    
    # Эталон в PyTorch
    t_linear = torch.nn.Linear(IN_FEAT, OUT_CH, bias=False)
    with torch.no_grad():
        t_linear.weight.copy_(torch.from_numpy(HostReturner.to_cpu(lin_layer.weight)))
        
    t_y_lin = t_linear(t_y_emb)
    
    # === ВНУТРИ ФАЙЛА tests/test_layers.py (Строка 104) ===
    # Выпрямляем 3D-выход PyTorch [2, 4, 8] в плоскую 2D-матрицу [8, 8] для честной сверки
    np.testing.assert_allclose(
        cl_res_lin, 
        t_y_lin.detach().numpy().reshape(-1, OUT_CH), 
        rtol=1e-4, 
        atol=1e-4
    )
    print("  ✅ Linear Forward: Проекция матриц через bmm_tiled сошлась с PyTorch!")
    
    print("\n🎉 ВСЕ ГЛОБАЛЬНЫЕ СЛОИ УСПЕШНО ВЕРИФИЦИРОВАНЫ!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_global_layers_pipeline()
