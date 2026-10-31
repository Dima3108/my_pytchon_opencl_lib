// Встроенные утилиты генерации шума и Dropout
inline float nn_rand(int tid, unsigned int seed) {
    unsigned int x = tid + seed;
    x = (x ^ 61) ^ (x >> 16);
    x += (x << 3);
    x = x ^ (x >> 4);
    x *= 0x27d4eb2d;
    x = x ^ (x >> 15);
    return (float)x / 4294967296.0f; 
}
inline float nn_dropout(float x, float p, int tid, unsigned int seed) {
    if (p <= 0.0f) return x;
    if (p >= 1.0f) return 0.0f;
    float r = nn_rand(tid, seed);
    return (r < p) ? 0.0f : x / (1.0f - p);
}
inline float nn_dropout_backward(float grad_out, float p, int tid, unsigned int seed) {
    float r = nn_rand(tid, seed);
    return (r < p) ? 0.0f : grad_out / (1.0f - p);
}

kernel void fused_elementwise_kernel(
    global float* A, const int A_len,
    global float* C, const int C_len,
    global float* B, const int B_len) {
int tid=get_global_size(1)*get_global_id(0)+get_global_id(1);

    // Восстановление многомерных координат из плоского tid
    int rem = tid;
    int dim_0 = rem;

    // Вычисление смещений с учетом правил Broadcasting
    int idx_A = 0;
    idx_A += dim_0 * 1;
    idx_A = min(idx_A, 100000 - 1);

    int idx_C = 0;
    idx_C += dim_0 * 1;
    idx_C = min(idx_C, 120000 - 1);

    int idx_B = 0;
    idx_B += dim_0 * 1;
    idx_B = min(idx_B, 80000 - 1);

    // Чтение во внутренние регистры
    float priv_A = A[idx_A];
    float priv_C = C[idx_C];
    float priv_B = B[idx_B];

    // Вычисления
    priv_A = tanh(priv_A);
    priv_C = cos(priv_B);
    priv_A = priv_A / 1.5f;
    priv_C = nn_dropout(priv_C, 0.2f, tid, 42u);
    priv_B = priv_B + 0.01f;

    // Сохранение результатов
    if (tid < A_len) { A[idx_A] = priv_A; }
    if (tid < C_len) { C[idx_C] = priv_C; }
    if (tid < B_len) { B[idx_B] = priv_B; }
}