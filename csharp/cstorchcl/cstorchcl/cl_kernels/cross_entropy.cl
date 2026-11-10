// Вход P (slicedLogits): [NumRows, VocabSize]
// Вход T (trueLossTarget): [NumRows, VocabSize] - One-Hot или вероятности
// Выход L (loss): [1] - скаляр ошибки

__kernel void cross_entropy_forward(
    __global const float* P, 
    __global const float* T, 
    __global float* L, 
    const int vocab_size,
    const int num_rows) 
{
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);
    
    __local float l_mem[256];

    // 1. Поиск максимума в строке логитов (Safe Log-Sum-Exp)
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += gsize) {
        float val = P[row * vocab_size + i];
        if (val > local_max) local_max = val;
    }
    l_mem[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (l_mem[tid + s] > l_mem[tid]) l_mem[tid] = l_mem[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = l_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Вычисление суммы экспонент
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += gsize) {
        local_sum += exp(P[row * vocab_size + i] - max_val);
    }
    l_mem[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_mem[tid] += l_mem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum_val = l_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Вычисление Cross-Entropy значения для этой строки
    // Формула: Loss = sum( -T_i * (P_i - max_val - log(sum_val)) )
    float log_sum_exp = max_val + log(sum_val);
    float row_loss = 0.0f;
    for (int i = tid; i < vocab_size; i += gsize) {
        int idx = row * vocab_size + i;
        if (T[idx] > 0.0f) { // Оптимизация под разреженный One-Hot
            row_loss += -T[idx] * (P[idx] - log_sum_exp);
        }
    }
    
    l_mem[tid] = row_loss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_mem[tid] += l_mem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Записываем среднее по всем строкам атомарно в финальный скаляр
    if (tid == 0) {
        // Так как это скаляр, один поток добавляет среднее значение своей строки
        // Используем встроенный трюк с атомиками или кастомный atomic_add_float из прошлых шагов
        // Для простоты, так как num_rows обычно небольшой (Batch * SeqLen), сделаем atomic_add
        // Если atomic_add_float объявлен глобально в вашей системе, вызываем его:
        union { unsigned int intVal; float floatVal; } next, expected, current;
        global volatile unsigned int* source = (global volatile unsigned int*)L;
        float val_to_add = l_mem[0] / num_rows;
        
        current.floatVal = *L;
        do {
            expected.floatVal = current.floatVal;
            next.floatVal = expected.floatVal + val_to_add;
            current.intVal = atom_cmpxchg(source, expected.intVal, next.intVal);
        } while (current.intVal != expected.intVal);
    }
}

// Ядро обратного прохода (Математически объединяет Softmax-градиент и CE-градиент)
// Формула: dL/dP_i = (Softmax(P_i) - T_i) / NumRows
__kernel void cross_entropy_backward(
    __global const float* P, 
    __global const float* T, 
    __global float* GI, 
    const int vocab_size,
    const int num_rows) 
{
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);
    
    __local float l_mem[256];

    // Шаг 1: Ищем максимум для стабильного Softmax внутри backward
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += gsize) {
        float val = P[row * vocab_size + i];
        if (val > local_max) local_max = val;
    }
    l_mem[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (l_mem[tid + s] > l_mem[tid]) l_mem[tid] = l_mem[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = l_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Шаг 2: Считаем сумму экспонент
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += gsize) {
        local_sum += exp(P[row * vocab_size + i] - max_val);
    }
    l_mem[tid] = local_sum;
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_mem[tid] += l_mem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum_val = l_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Шаг 3: Вычисляем финальный градиент по входу
    for (int i = tid; i < vocab_size; i += gsize) {
        int idx = row * vocab_size + i;
        float softmax_val = exp(P[idx] - max_val) / sum_val;
        
        // (Softmax - Target) / Общее_кол-во_элементов
        GI[idx] = (softmax_val - T[idx]) / num_rows;
    }
}
