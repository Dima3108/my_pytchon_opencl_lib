// Распаковка линейного индекса в координаты по заданной форме
inline void get_coords_permute(int linear_idx, __global const int* dims, int num_dims, int* out_coords) {
    int rem = linear_idx;
    for (int i = num_dims - 1; i >= 0; i--) {
        out_coords[i] = rem % dims[i];
        rem /= dims[i];
    }
}

// Кернел универсальной перестановки осей тензора любой размерности (до 10D)
__kernel void tensor_permute(
    __global const float* src,
    __global float* dst,
    __global const int* src_dims,
    __global const int* dst_dims,
    __global const int* permute_axes, // Пример для x.transpose(1, 2): [0, 2, 1, 3]
    const int num_dims,
    const int total_elements)
{
    int tid = get_global_id(0);
    if (tid >= total_elements) return;

    int dst_coords[10];
    int src_coords[10];

    // 1. Получаем многомерные координаты текущего потока в ВЫХОДНОМ тензоре
    get_coords_permute(tid, dst_dims, num_dims, dst_coords);

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        if (i < num_dims) {
            // 2. Переставляем оси обратно, чтобы узнать координаты в ИСХОДНОМ тензоре
            // Исходная координата на позиции permute_axes[i] равна текущей выходной координате i
            src_coords[permute_axes[i]] = dst_coords[i];
        }
    }

    
    // 3. Собираем линейный индекс в исходном тензоре src
    int src_idx_1 = 0;
    int stride = 1;
    #pragma unroll
    for (int i = num_dims - 1; i >= 0; i--) {
        src_idx_1 += src_coords[i] * stride;
        stride *= src_dims[i];
    }

    // 4. Переносим элемент
    dst[tid] = src[src_idx_1];
}
