// Ядро прямого прохода: Суммирование по произвольной оси
// Global Work Size: [outer_size, inner_size]
__kernel void reduce_sum_axis(
    __global const float* X,
    __global float* Y,
    const int outer_size,
    const int axis_size,
    const int inner_size)
{
    int outer_idx = get_global_id(0);
    int inner_idx = get_global_id(1);

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float sum = 0.0f;
    // Базовое смещение в исходном тензоре для текущего среза
    int base_offset = outer_idx * (axis_size * inner_size) + inner_idx;

    for (int a = 0; a < axis_size; a++) {
        sum += X[base_offset + a * inner_size];
    }

    Y[outer_idx * inner_size + inner_idx] = sum;
}

// Ядро обратного прохода: Растягивание (Broadcast) градиента обратно по оси суммирования
// Математически производная суммы d(sum(X))/dX_i = 1.0f для каждого элемента оси.
// Поэтому пришедший сверху градиент gradOutput просто копируется во все ячейки вдоль оси.
// Global Work Size: [outer_size, inner_size]
__kernel void reduce_sum_axis_backward(
    __global const float* grad_output,
    __global float* grad_input,
    const int outer_size,
    const int axis_size,
    const int inner_size)
{
    int outer_idx = get_global_id(0);
    int inner_idx = get_global_id(1);

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float grad_val = grad_output[outer_idx * inner_size + inner_idx];
    int base_offset = outer_idx * (axis_size * inner_size) + inner_idx;

    for (int a = 0; a < axis_size; a++) {
        grad_input[base_offset + a * inner_size] = grad_val;
    }
}
