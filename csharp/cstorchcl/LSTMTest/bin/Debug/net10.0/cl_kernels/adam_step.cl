kernel void adam_step(
    global float* weight,
    global const float* grad,
    global float* m,
    global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float bias_correction1,
    const float bias_correction2,
    const int total_elements) 
{
    int id = get_global_id(0);
    if (id >= total_elements) return;

    float g = grad[id];

    // 1. Обновление первого момента (бегущее среднее градиента)
    float m_val = beta1 * m[id] + (1.0f - beta1) * g;
    m[id] = m_val;

    // 2. Обновление второго момента (бегущее среднее квадрата градиента)
    float v_val = beta2 * v[id] + (1.0f - beta2) * (g * g);
    v[id] = v_val;

    // 3. Вычисление несмещенных оценок моментов
    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;

    // 4. Обновление веса модели
    weight[id] -= (lr / (sqrt(v_hat) + epsilon)) * m_hat;
}
