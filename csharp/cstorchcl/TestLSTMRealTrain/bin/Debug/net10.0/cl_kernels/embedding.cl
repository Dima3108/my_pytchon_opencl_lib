// Кастомная атомарная функция для float, совместимая со спецификацией OpenCL 1.2
inline void atomic_add_float(__global volatile float* address, const float val) {
    // Трактуем указатель на float как указатель на int для работы атомика
    __global volatile int* int_address = (__global volatile int*)address;
    
    int old_val, new_val, expected_val;
    
    do {
        old_val = *int_address;
        // Превращаем битовое представление int обратно во float, прибавляем val
        // и конвертируем итоговый float обратно в биты int
        float sum = as_float(old_val) + val;
        new_val = as_int(sum);
        
        expected_val = old_val;
        // Пытаемся записать новое значение, если ячейка памяти не была изменена другим потоком
        old_val = atom_cmpxchg(int_address, expected_val, new_val);
        
    } while (old_val != expected_val); // Повторяем, если запись сорвалась из-за коллизии
}

// ============================================================================
// ПРЯМОЙ ПРОХОД EMBEDDING (ПРАВИЛЬНЫЙ ПОРЯДОК ОСЕЙ ДЛЯ COALESCED ACCESS)
// Сетка запуска: Global = [embedding_dim, total_tokens], Local = null (выберет драйвер)
// ============================================================================
kernel void embedding_forward(
    global const int* X_tokens,      // [total_tokens]
    global const float* weight,      // [vocab_size, embedding_dim]
    global float* Y_out,             // [total_tokens, embedding_dim]
    const int embedding_dim,
    const int total_tokens
) {
    // Ось 0 (быстрая) -> dim_idx (соседние потоки читают/пишут соседние float!)
    int dim_idx = get_global_id(0);
    // Ось 1 -> token_idx
    int token_idx = get_global_id(1);

    if (dim_idx >= embedding_dim || token_idx >= total_tokens) return;

    int token_id = X_tokens[token_idx];
    Y_out[token_idx * embedding_dim + dim_idx] = weight[token_id * embedding_dim + dim_idx];
}

// ============================================================================
// ОБРАТНЫЙ ПРОХОД EMBEDDING (БЕЗАТОМАРНЫЙ GATHER + КЭШ ТОКЕНОВ В __local)
// Сетка запуска: Global = [padded_dim, vocab_size], Local = [64, 4] (256 потоков)
// ============================================================================
#define EMB_TILE_TOKENS 256

kernel void embedding_backward(
    global const float* grad_output, // [total_tokens, embedding_dim]
    global const int* X_tokens,      // [total_tokens]
    global float* grad_weight,       // [vocab_size, embedding_dim]
    const int embedding_dim,
    const int total_tokens,
    const int vocab_size
) {
    int dim_idx = get_global_id(0);
    int vocab_id = get_global_id(1);

    int flat_lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int wg_size = get_local_size(0) * get_local_size(1);

    // Кэшируем порцию ID токенов в быстрой локальной памяти группы (всего 1 КБ)
    __local int local_tokens[EMB_TILE_TOKENS];

    float sum = 0.0f;

    for (int base_t = 0; base_t < total_tokens; base_t += EMB_TILE_TOKENS) {
        // Кооперативная загрузка 256 токенов в __local память всеми потоками группы
        for (int i = flat_lid; i < EMB_TILE_TOKENS; i += wg_size) {
            int t_idx = base_t + i;
            local_tokens[i] = (t_idx < total_tokens) ? X_tokens[t_idx] : -1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (dim_idx < embedding_dim && vocab_id < vocab_size) {
            int limit = min(EMB_TILE_TOKENS, total_tokens - base_t);
            for (int i = 0; i < limit; i++) {
                if (local_tokens[i] == vocab_id) {
                    sum += grad_output[(base_t + i) * embedding_dim + dim_idx];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Прямая запись без атомиков и без предварительного обнуления буфера на CPU!
    if (dim_idx < embedding_dim && vocab_id < vocab_size) {
        grad_weight[vocab_id * embedding_dim + dim_idx] = sum;
    }
}
