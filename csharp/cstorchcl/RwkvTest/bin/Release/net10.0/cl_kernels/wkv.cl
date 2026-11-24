// cl_kernels/wkv.cl

kernel void wkv_forward(
    global const float* w,      // [C] (затухание, обычно отрицательное: -exp(w_raw))
    global const float* u,      // [C] (бонус текущего токена time_first)
    global const float* k,      // [B, T, C]
    global const float* v,      // [B, T, C]
    global float* y,            // [B, T, C] (выход)
    global float* state_aa,     // [B, T, C] (кэш числителя для backward)
    global float* state_bb,     // [B, T, C] (кэш знаменателя для backward)
    global float* state_pp,     // [B, T, C] (кэш максимума экспоненты для backward)
    const int B,
    const int T,
    const int C
) {
    int idx = get_global_id(0);
    if (idx >= B * C) return;

    int b = idx / C;
    int c = idx % C;

    float _w = w[c];
    float _u = u[c];

    float aa = 0.0f;
    float bb = 0.0f;
    float pp = -1e38f; // -inf

    for (int t = 0; t < T; t++) {
        int offset = b * T * C + t * C + c;
        float kk = k[offset];
        float vv = v[offset];

        // Сохраняем состояние ДО обновления (нужно для backward)
        state_aa[offset] = aa;
        state_bb[offset] = bb;
        state_pp[offset] = pp;

        // 1. Вычисляем выход y_t с учетом бонуса u
        float ww = _u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[offset] = (e1 * aa + e2 * vv) / (e1 * bb + e2);

        // 2. Обновляем рекуррентное состояние с затуханием w
        ww = _w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
}

kernel void wkv_backward(
    global const float* w,          // [C]
    global const float* u,          // [C]
    global const float* k,          // [B, T, C]
    global const float* v,          // [B, T, C]
    global const float* y,          // [B, T, C]
    global const float* gy,         // [B, T, C] (gradOutput)
    global const float* state_aa,   // [B, T, C]
    global const float* state_bb,   // [B, T, C]
    global const float* state_pp,   // [B, T, C]
    global float* gw_part,          // [B, C] (частичный градиент по w для батча b)
    global float* gu_part,          // [B, C] (частичный градиент по u для батча b)
    global float* gk,               // [B, T, C]
    global float* gv,               // [B, T, C]
    const int B,
    const int T,
    const int C
) {
    int idx = get_global_id(0);
    if (idx >= B * C) return;

    int b = idx / C;
    int c = idx % C;

    float _w = w[c];
    float _u = u[c];

    float gw = 0.0f;
    float gu = 0.0f;

    // Обратные состояния (аккумуляторы градиентов из будущего)
    float gaa = 0.0f;
    float gbb = 0.0f;
    float gpp = -1e38f;

    for (int t = T - 1; t >= 0; t--) {
        int offset = b * T * C + t * C + c;
        float kk = k[offset];
        float vv = v[offset];
        float yy = y[offset];
        float gyy = gy[offset];

        float aa = state_aa[offset];
        float bb = state_bb[offset];
        float pp = state_pp[offset];

        // Восстанавливаем промежуточные значения forward-шага
        float ww = _u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        float denom = 1.0f / (e1 * bb + e2);

        // Градиенты от текущего выхода y_t
        float d_num = gyy * denom;
        float d_den = -gyy * yy * denom;

        float d_e2 = d_num * vv + d_den;
        float d_e1 = d_num * aa + d_den * bb;

        gu += d_e2 * e2;
        float dk_curr = d_e2 * e2;
        float dv_curr = d_num * e2;

        // Градиенты, пришедшие из будущих шагов времени (t + 1 ... T - 1)
        float ww_next = _w + pp;
        float p_next = max(ww_next, kk);
        float e1_n = exp(ww_next - p_next);
        float e2_n = exp(kk - p_next);

        // Масштабируем накопленные gaa, gbb к текущей экспоненте p_next
        float scale = exp(gpp - p_next);
        float gaa_scaled = gaa * scale;
        float gbb_scaled = gbb * scale;

        gw += (gaa_scaled * aa + gbb_scaled * bb) * e1_n;
        dk_curr += (gaa_scaled * vv + gbb_scaled) * e2_n;
        dv_curr += gaa_scaled * e2_n;

        gk[offset] = dk_curr;
        gv[offset] = dv_curr;

        // Обновляем gaa, gbb, gpp для шага t - 1
        float next_gaa = d_num * e1 + gaa_scaled * e1_n;
        float next_gbb = d_den * e1 + gbb_scaled * e1_n;
        
        // Перенормировка gpp для предотвращения переполнения
        float max_g = max(fabs(next_gaa), fabs(next_gbb));
        if (max_g > 0.0f) {
            float log_g = log(max_g);
            gaa = next_gaa / max_g;
            gbb = next_gbb / max_g;
            gpp = p_next + log_g; // сдвиг показателя
        } else {
            gaa = 0.0f;
            gbb = 0.0f;
            gpp = -1e38f;
        }
    }

    gw_part[idx] = gw;
    gu_part[idx] = gu;
}
