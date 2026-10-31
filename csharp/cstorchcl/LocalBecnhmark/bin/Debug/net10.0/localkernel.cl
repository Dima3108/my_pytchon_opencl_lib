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
    int tid = get_global_id(0);
    float priv_A = A[min(tid, A_len - 1)];
    float priv_C = C[min(tid, C_len - 1)];
    float priv_B = B[min(tid, B_len - 1)];

    // Вычисления
    priv_A = tanh(priv_A);
    priv_C = cos(priv_B);
    priv_A = priv_A / 1.5f;
    priv_C = nn_dropout(priv_C, 0.2f, tid, 42u);
    priv_B = priv_B + 0.01f;

    if (tid < A_len) { A[tid] = priv_A; }
    if (tid < C_len) { C[tid] = priv_C; }
    if (tid < B_len) { B[tid] = priv_B; }
}