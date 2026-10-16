// Ядро обновления весов методом SGD
// W: data_buffer параметра (весов или смещений)
// G: grad_buffer параметра
__kernel void sgd_step_kernel(
    __global float* W,
    __global const float* G,
    const int size,
    const float lr,
    const float weight_decay,
    const int warp_size)
{
    // Наш универсальный автоматический 2D-индекс потока
    int tid = get_global_id(0) * warp_size + get_global_id(1);
    if (tid >= size) return;

    float w_val = W[tid];
    float g_val = G[tid];

    // Если включен weight decay, добавляем штраф к градиенту (L2 регуляризация)
    if (weight_decay > 0.0f) {
        g_val += weight_decay * w_val;
    }

    // Обновляем вес на месте во VRAM
    W[tid] = w_val - lr * g_val;
}
// Ядро обновления весов методом AdamW
__kernel void adamw_step_kernel(
    __global float* W,                  // Буфер весов (data_buffer)
    __global const float* G,            // Буфер градиентов (grad_buffer)
    __global float* M,                  // Буфер первого момента (momentum)
    __global float* V,                  // Буфер второго момента (velocity)
    const int size,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float bias_correction1,       // 1.0f - pow(beta1, step)
    const float bias_correction2,       // 1.0f - pow(beta2, step)
    const int warp_size)
{
    int tid = get_global_id(0) * warp_size + get_global_id(1);
    if (tid >= size) return;

    float w_val = W[tid];
    float g_val = G[tid];

    // 1. Отделенный Weight Decay (L2 регуляризация применяется прямо к весу)
    w_val -= lr * weight_decay * w_val;

    // 2. Скользящее среднее моментов градиентов
    float m_val = beta1 * M[tid] + (1.0f - beta1) * g_val;
    float v_val = beta2 * V[tid] + (1.0f - beta2) * g_val * g_val;

    // Перезаписываем состояния моментов во VRAM на месте
    M[tid] = m_val;
    V[tid] = v_val;

    // 3. Коррекция смещения для первых шагов обучения
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;

    // 4. Финальное инкрементальное обновление веса
    W[tid] = w_val - (lr / (sqrt(v_hat) + epsilon)) * m_hat;
}
