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


// Добавляем к строке utilities в LocalRoslynTranspiler:

inline float nn_relu(float x) { return (x > 0.0f) ? x : 0.0f; }
inline float nn_relu_backward(float x) { return (x > 0.0f) ? 1.0f : 0.0f; }

inline float nn_leakyrelu(float x) { return (x > 0.0f) ? x : x * 0.01f; }
inline float nn_leakyrelu_backward(float x) { return (x > 0.0f) ? 1.0f : 0.01f; }

inline float nn_softplus(float x) { return log1p(exp(x)); }
inline float nn_softplus_backward(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float nn_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float nn_sigmoid_backward(float x) { 
    float s = 1.0f / (1.0f + exp(-x)); 
    return s * (1.0f - s); 
}

inline float nn_tanh_backward(float x) { 
    float t = tanh(x); 
    return 1.0f - t * t; 
}




kernel void fused_elementwise_backward_kernel(
    global float* P, global float* grad_P, global float* out_grad_P, const int P_len,
    global float* b, global float* grad_b, global float* out_grad_b, const int b_len) {
int tid=get_global_size(1)*get_global_id(0)+get_global_id(1);

    // Восстановление многомерных координат из плоского tid
    int rem = tid;
    int dim_2 = rem % 6565;
    rem /= 6565;
    int dim_1 = rem % 1500;
    rem /= 1500;
    int dim_0 = rem;

    // Вычисление смещений с учетом правил Broadcasting
    int idx_P = 0;
    idx_P += dim_0 * 9847500;
    idx_P += dim_1 * 6565;
    idx_P += dim_2 * 1;
    idx_P = min(idx_P, 78780000 - 1);

    int idx_b = 0;
    idx_b += 0 * 6565;
    idx_b += dim_2 * 1;
    idx_b = min(idx_b, 6565 - 1);

    float priv_P = P[idx_P];
    float priv_grad_P = grad_P[idx_P];
    float priv_out_grad_P = 0.0f;
    float priv_b = b[idx_b];
    float priv_grad_b = grad_b[idx_b];
    float priv_out_grad_b = 0.0f;

    // Вычисление производных на регистрах
    priv_out_grad_P += priv_grad_P;
    priv_out_grad_b += priv_grad_P;

    // Сохранение рассчитанных градиентов
    if (tid < P_len) { out_grad_P[idx_P] = priv_out_grad_P; }
    if (tid < b_len) { out_grad_b[idx_b] = priv_out_grad_b; }
}