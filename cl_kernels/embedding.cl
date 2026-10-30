// Кастомная атомарная функция для float, совместимая со спецификацией OpenCL 1.2
inline void atomic_add_float(__global volatile float* address, const float val) 
{
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

// Forward шаг: Выборка эмбеддингов по ID токенов
__kernel void embedding_forward(
    __global const int* X_tokens,
    __global const float* weight,
    __global float* Y_out,
    const int embedding_dim,
    const int total_tokens)
{
    int token_idx = get_global_id(0);
    int dim_idx = get_global_id(1);

    if (token_idx >= total_tokens || dim_idx >= embedding_dim) return;

    int token_id = X_tokens[token_idx];
    int weight_idx = token_id * embedding_dim + dim_idx;
    int out_idx = token_idx * embedding_dim + dim_idx;

    Y_out[out_idx] = weight[weight_idx];
}

// Backward шаг: Безопасная аккумуляция градиентов по матрице весов
__kernel void embedding_backward(
    __global const float* grad_output,
    __global const int* X_tokens,
    __global float* grad_weight,
    const int embedding_dim,
    const int total_tokens)
{
    int token_idx = get_global_id(0);
    int dim_idx = get_global_id(1);

    if (token_idx >= total_tokens || dim_idx >= embedding_dim) return;

    int token_id = X_tokens[token_idx];
    int go_idx = token_idx * embedding_dim + dim_idx;
    int gw_idx = token_id * embedding_dim + dim_idx;

    // Вызываем нашу кастомную атомарную функцию вместо стандартного atom_add
    atomic_add_float(&grad_weight[gw_idx], grad_output[go_idx]);
}

