// BLOCK_SIZE — количество потоков в рабочей группе (например, 256)
// D_MODEL — размерность скрытого состояния (например, 400 или 4096)

__kernel void rmsnorm_forward(
    __global const float* X,       // Входной тензор [B * L, D_MODEL]
    __global const float* gamma,   // Обучаемые веса [D_MODEL]
    __global float* Y,             // Выходной тензор [B * L, D_MODEL]
    __global float* inv_rms_cache, // Кэш для Backward: 1/RMS для каждой строки [B * L]
    const int d_model) 
{
    int row = get_group_id(0);     // Индекс строки (токен в батче)
    int tid = get_local_id(0);    // Индекс потока внутри группы
    int gsize = get_local_size(0); // Количество потоков в группе

    // Локальный массив для редукции суммы квадратов
    __local float l_sum[256]; // Размер должен быть равен BLOCK_SIZE

    // 1. Каждая группа потоков параллельно считает сумму квадратов элементов строки
    float local_sq_sum = 0.0f;
    for (int i = tid; i < d_model; i += gsize) {
        float val = X[row * d_model + i];
        local_sq_sum += val * val;
    }
    l_sum[tid] = local_sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Древовидная редукция внутри локальной памяти
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_sum[tid] += l_sum[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Поток 0 вычисляет финальный инвертированный RMS (Root Mean Square)
    __local float inv_rms;
    if (tid == 0) {
        // Добавляем epsilon (1e-5f) для стабильности
        inv_rms = 1.0f / sqrt(l_sum[0] / d_model + 1e-5f);
        inv_rms_cache[row] = inv_rms; // Сохраняем в глобальную память для Backward
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Нормализация элементов и умножение на веса gamma
    for (int i = tid; i < d_model; i += gsize) {
        int idx = row * d_model + i;
        Y[idx] = (X[idx] * inv_rms) * gamma[i];
    }
}
__kernel void rmsnorm_backward(
    __global const float* grad_output, // Пришедший градиент dL/dY [B * L, D_MODEL]
    __global const float* X,           // Исходный вход X [B * L, D_MODEL]
    __global const float* gamma,       // Веса [D_MODEL]
    __global const float* inv_rms_cache,// Сохраненный инвертированный RMS [B * L]
    __global float* grad_input,        // Выходной градиент dL/dX [B * L, D_MODEL]
    __global float* grad_gamma_part,   // Промежуточный градиент по весам (для редукции)
    const int d_model) 
{
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);

    float inv_rms = inv_rms_cache[row];
    
    __local float l_sum[256];

    // Вычисляем общую часть градиента: sum(grad_output * gamma * X)
    float local_dot = 0.0f;
    for (int i = tid; i < d_model; i += gsize) {
        int idx = row * d_model + i;
        float x_norm = X[idx] * inv_rms;
        
        // Попутно поток может копить градиент по весам gamma для текущей строки
        // grad_gamma_part[row * d_model + i] = grad_output[idx] * x_norm;

        local_dot += grad_output[idx] * gamma[i] * X[idx];
    }
    l_sum[tid] = local_dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Редукция суммы
    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) l_sum[tid] += l_sum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float sum_grad_gamma_x = l_sum[0];

    // Математическая формула градиента RMSNorm по входу X:
    // dL/dX = (inv_rms / d_model) * (d_model * grad_output * gamma - X * inv_rms^2 * sum(grad_output * gamma * X))
    float common_scale = inv_rms / d_model;
    float projection_scale = inv_rms * inv_rms * sum_grad_gamma_x;

    for (int i = tid; i < d_model; i += gsize) {
        int idx = row * d_model + i;
        grad_input[idx] = common_scale * (d_model * grad_output[idx] * gamma[i] - X[idx] * projection_scale);
    }
}
