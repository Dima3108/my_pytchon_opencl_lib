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

// Замените wkv_backward в cl_kernels/wkv.cl на эту версию:
kernel void wkv_backward(
    global const float* w,          // [C] (уже отрицательное: -exp(w_raw))
    global const float* u,          // [C]
    global const float* k,          // [B, T, C]
    global const float* v,          // [B, T, C]
    global const float* y,          // [B, T, C]
    global const float* gy,         // [B, T, C]
    global const float* state_aa,   // [B, T, C]
    global const float* state_bb,   // [B, T, C]
    global const float* state_pp,   // [B, T, C]
    global float* gw_part,          // [B, C]
    global float* gu_part,          // [B, C]
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

    // Градиенты по состоянию (aa, bb) в шкале текущего шага
    float gaa = 0.0f;
    float gbb = 0.0f;

    for (int t = T - 1; t >= 0; t--) {
        int offset = b * T * C + t * C + c;
        float kk = k[offset];
        float vv = v[offset];
        float yy = y[offset];
        float gyy = gy[offset];

        float aa = state_aa[offset];
        float bb = state_bb[offset];
        float pp = state_pp[offset];

        // 1. Восстанавливаем прямой проход выхода y_t
        float ww = _u + kk;
        float p = max(pp, ww);
        float e1 = exp(clamp(pp - p, -80.0f, 0.0f));
        float e2 = exp(clamp(ww - p, -80.0f, 0.0f));
        float denom = 1.0f / (e1 * bb + e2 + 1e-12f);

        float d_num = gyy * denom;
        float d_den = -gyy * yy * denom;

        float d_e2 = d_num * vv + d_den;
        gu += d_e2 * e2;

        float dk_curr = d_e2 * e2;
        float dv_curr = d_num * e2;

        // 2. Учитываем вклад от будущих шагов (gaa, gbb находятся в шкале p_next)
        float ww_next = _w + pp;
        float p_next = max(ww_next, kk);
        float e1_n = exp(clamp(ww_next - p_next, -80.0f, 0.0f));
        float e2_n = exp(clamp(kk - p_next, -80.0f, 0.0f));

        gw += (gaa * aa + gbb * bb) * e1_n;
        dk_curr += (gaa * vv + gbb) * e2_n;
        dv_curr += gaa * e2_n;

        gk[offset] = dk_curr;
        gv[offset] = dv_curr;

        // 3. Переносим gaa и gbb на шаг t - 1 (в шкалу pp = state_pp[t])
        // Обратите внимание:
        // d(y_t)/d(aa_t) = e1 * d_num, где aa_t масштабирован под pp.
        // Из будущего: aa_{t+1} = e1_n * aa_t + e2_n * vv, где e1_n = exp(_w + pp - p_next).
        // Значит d(L)/d(aa_t) = d_num * e1 + gaa * e1_n — УЖЕ В ШКАЛЕ pp!
        gaa = d_num * e1 + gaa * e1_n;
        gbb = d_den * e1 + gbb * e1_n;
    }

    gw_part[idx] = gw;
    gu_part[idx] = gu;
}

