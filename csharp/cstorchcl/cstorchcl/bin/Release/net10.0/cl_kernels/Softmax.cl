__kernel void softmax_forward(
    __global const float* X, // Входная матрица (например, скоры внимания) [B * H * L, SEQ_LEN]
    __global float* Y,       // Выходной тензор вероятностей [B * H * L, SEQ_LEN]
    const int seq_len) 
{
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);

    __local float l_mem[256]; // Общая локальная память под редукции максимума и суммы

    // 1. Поиск максимума строки (Safe Softmax)
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += gsize) {
        float val = X[row * seq_len + i];
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
    barrier(CLK_LOCAL_MEM_FENCE); // Переиспользуем локальную память

    // 2. Вычисление экспонент и их суммы
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += gsize) {
        int idx = row * seq_len + i;
        float e = exp(X[idx] - max_val);
        Y[idx] = e; // Временно пишем экспоненту в выходной массив
        local_sum += e;
    }
    l_mem[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_mem[tid] += l_mem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum_val = l_mem[0];

    // 3. Финальное деление на сумму
    for (int i = tid; i < seq_len; i += gsize) {
        int idx = row * seq_len + i;
        Y[idx] /= sum_val; 
    }
}
__kernel void softmax_backward(
    __global const float* grad_output, // Градиент dL/dY [B * H * L, SEQ_LEN]
    __global const float* Y,           // Исходный ВЫХОД Softmax (вероятности) [B * H * L, SEQ_LEN]
    __global float* grad_input,        // Градиент dL/dX [B * H * L, SEQ_LEN]
    const int seq_len) 
{
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);

    __local float l_sum[256];

    // Формула: dL/dX_i = Y_i * (grad_output_i - sum(grad_output_j * Y_j))
    // Сначала считаем скалярное произведение grad_output и Y по строке
    float local_dot = 0.0f;
    for (int i = tid; i < seq_len; i += gsize) {
        int idx = row * seq_len + i;
        local_dot += grad_output[idx] * Y[idx];
    }
    l_sum[tid] = local_dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Редукция суммы скалярного произведения
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_sum[tid] += l_sum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum_grad_times_y = l_sum[0];

    // Вычисляем финальный градиент по входу
    for (int i = tid; i < seq_len; i += gsize) {
        int idx = row * seq_len + i;
        grad_input[idx] = Y[idx] * (grad_output[idx] - sum_grad_times_y);
    }
}
