// Быстрый генератор псевдослучайных чисел [0.0f, 1.0f) на основе хэша PCG
inline float hash_rng(uint idx, uint seed)
{
    uint state = idx * 747796405u + seed * 2891336453u + 277803737u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    // Преобразуем 24 старших бита во float [0, 1)
    return (float)(word >> 8u) * (1.0f / 16777216.0f);
}

// =========================================================================
// FORWARD: Inverted Dropout с записью 1-байтной маски
// =========================================================================
__kernel void dropout_forward(
    __global const float* X,
    __global float* Y,
    __global uchar* mask,
    const float p,
    const float scale,
    const uint seed,
    const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    float r = hash_rng((uint)idx, seed);
    if (r >= p) {
        Y[idx] = X[idx] * scale;
        mask[idx] = 1;
    } else {
        Y[idx] = 0.0f;
        mask[idx] = 0;
    }
}

// =========================================================================
// BACKWARD: Пропускаем градиент только через выжившие нейроны
// =========================================================================
__kernel void dropout_backward(
    __global const float* dY,
    __global const uchar* mask,
    __global float* dX,
    const float scale,
    const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    dX[idx] = (mask[idx] != 0) ? (dY[idx] * scale) : 0.0f;
}
