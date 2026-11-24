// cl_kernels/rwkv_ops.cl

// 1. TokenShift: out[b, t, c] = mu[c] * x[b, t, c] + (1 - mu[c]) * (t > 0 ? x[b, t-1, c] : 0)
kernel void token_shift_forward(
    global const float* x,      // [B, T, C]
    global const float* mu,     // [C]
    global float* out,          // [B, T, C]
    const int B, const int T, const int C
) {
    int idx = get_global_id(0);
    if (idx >= B * T * C) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    float m = mu[c];
    float curr = x[idx];
    float prev = (t > 0) ? x[b * T * C + (t - 1) * C + c] : 0.0f;

    out[idx] = m * curr + (1.0f - m) * prev;
}

kernel void token_shift_backward(
    global const float* x,          // [B, T, C]
    global const float* mu,         // [C]
    global const float* grad_out,   // [B, T, C]
    global float* grad_x,           // [B, T, C]
    global float* grad_mu_part,     // [B * T, C] (частичный градиент для редукции по строкам)
    const int B, const int T, const int C
) {
    int idx = get_global_id(0);
    if (idx >= B * T * C) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    float m = mu[c];
    float go = grad_out[idx];

    // Градиент по x[b, t, c] приходит от шага t (с весом mu) и от шага t+1 (с весом 1 - mu)
    float gx = go * m;
    if (t + 1 < T) {
        float go_next = grad_out[b * T * C + (t + 1) * C + c];
        gx += go_next * (1.0f - m);
    }
    grad_x[idx] = gx;

    // Градиент по mu[c]: go * (x[t] - x[t-1])
    float curr = x[idx];
    float prev = (t > 0) ? x[b * T * C + (t - 1) * C + c] : 0.0f;
    grad_mu_part[idx] = go * (curr - prev);
}

// 2. Активации RWKV:
// mode == 0: Sigmoid: y = 1 / (1 + exp(-x))
// mode == 1: Squared ReLU: y = max(0, x)^2
// mode == 2: NegExp (для w_decay): y = -exp(x)
kernel void rwkv_unary_forward(
    global const float* x,
    global float* y,
    const int size,
    const int mode
) {
    int id = get_global_id(0);
    if (id >= size) return;

    float val = x[id];
    if (mode == 0) {
        y[id] = 1.0f / (1.0f + exp(-val));
    } else if (mode == 1) {
        float r = max(0.0f, val);
        y[id] = r * r;
    } else if (mode == 2) {
        y[id] = -exp(val);
    }
}

kernel void rwkv_unary_backward(
    global const float* x,
    global const float* y,
    global const float* grad_y,
    global float* grad_x,
    const int size,
    const int mode
) {
    int id = get_global_id(0);
    if (id >= size) return;

    float gy = grad_y[id];
    if (mode == 0) {
        float s = y[id];
        grad_x[id] = gy * s * (1.0f - s);
    } else if (mode == 1) {
        float val = x[id];
        grad_x[id] = (val > 0.0f) ? (gy * 2.0f * val) : 0.0f;
    } else if (mode == 2) {
        // y = -exp(x) => dy/dx = -exp(x) = y
        grad_x[id] = gy * y[id];
    }
}

// 3. Поэлементное умножение двух тензоров (Hadamard product: z = a * b)
kernel void tensor_mul_forward(
    global const float* a,
    global const float* b,
    global float* c,
    const int size
) {
    int id = get_global_id(0);
    if (id >= size) return;
    c[id] = a[id] * b[id];
}

kernel void tensor_mul_backward(
    global const float* a,
    global const float* b,
    global const float* grad_c,
    global float* grad_a,
    global float* grad_b,
    const int size
) {
    int id = get_global_id(0);
    if (id >= size) return;
    float gc = grad_c[id];
    grad_a[id] = gc * b[id];
    grad_b[id] = gc * a[id];
}
