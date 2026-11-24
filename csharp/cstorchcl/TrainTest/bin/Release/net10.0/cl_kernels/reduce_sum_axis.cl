// cl_kernels/reduce_sum_axis.cl

__kernel void reduce_sum_axis(
    __global const float* X,
    __global float* Y,
    const int outer_size,
    const int axis_size,
    const int inner_size
) {
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int wg_x = get_local_size(0); // 32 (если inner_size >= 32) или 1
    int wg_y = get_local_size(1); // 8  (если inner_size >= 32) или 256

    int inner_idx = get_group_id(0) * wg_x + lx;
    int outer_idx = get_group_id(1);

    __local float sdata[256];
    int tid = ly * wg_x + lx;

    float sum = 0.0f;
    if (outer_idx < outer_size && inner_idx < inner_size) {
        int base_offset = outer_idx * (axis_size * inner_size) + inner_idx;
        for (int a = ly; a < axis_size; a += wg_y) {
            sum += X[base_offset + a * inner_size];
        }
    }
    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Древовидная редукция по оси Y
    for (int stride = wg_y >> 1; stride > 0; stride >>= 1) {
        if (ly < stride) {
            sdata[tid] += sdata[(ly + stride) * wg_x + lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (ly == 0 && outer_idx < outer_size && inner_idx < inner_size) {
        Y[outer_idx * inner_size + inner_idx] = sdata[lx];
    }
}

// Для Backward локальная память не нужна: убираем последовательный цикл,
// делая плоскую 1D-сетку по всем элементам выходного тензора (100% coalesced write)
__kernel void reduce_sum_axis_backward(
    __global const float* grad_output,
    __global float* grad_input,
    const int total_elements,
    const int axis_size,
    const int inner_size
) {
    int id = get_global_id(0);
    if (id >= total_elements) return;

    int inner_idx = id % inner_size;
    int outer_idx = id / (axis_size * inner_size);
    grad_input[id] = grad_output[outer_idx * inner_size + inner_idx];
}
