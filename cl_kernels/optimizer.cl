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
