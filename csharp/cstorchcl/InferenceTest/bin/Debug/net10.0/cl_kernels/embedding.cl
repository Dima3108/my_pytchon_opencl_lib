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

#define EMB_TILE_TOKENS 256

kernel void embedding_backward(
    global const float* grad_output, // [total_tokens, embedding_dim]
    global const int* X_tokens,      // [total_tokens]
    global float* grad_weight,       // [vocab_size, embedding_dim]
    const int embedding_dim,
    const int total_tokens,
    const int vocab_size
) {
    int dim_idx  = get_global_id(0); // [0 .. embedding_dim - 1] -> быстрая коалесцированная ось
    int vocab_id = get_global_id(1); // [0 .. vocab_size - 1]

    // Теперь барьеров нет, поэтому ранний выход абсолютно безопасен!
    if (dim_idx >= embedding_dim || vocab_id >= vocab_size) return;

    float sum = 0.0f;
    global const float* grad_ptr = grad_output + dim_idx;

    #pragma unroll 8
    for (int t = 0; t < total_tokens; t++) {
        // Аппаратный Broadcast из L1-кэша: все потоки варпа читают один и тот же X_tokens[t]
        int token = X_tokens[t];

        // Читаем градиент только при совпадении токена (избегаем чтения 12000 float для редких слов!)
        if (token == vocab_id) {
            sum += grad_ptr[t * embedding_dim];
        }
    }

    grad_weight[vocab_id * embedding_dim + dim_idx] = sum;
}

