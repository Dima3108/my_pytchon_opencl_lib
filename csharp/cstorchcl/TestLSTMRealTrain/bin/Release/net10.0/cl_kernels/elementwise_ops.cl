// Превращает линейный индекс потока во многомерные координаты выходного тензора
inline void get_coords(int linear_idx, __global const int* dims, int num_dims, int* out_coords) {
    int rem = linear_idx;
    for (int i = num_dims - 1; i >= 0; i--) {
        out_coords[i] = rem % dims[i];
        rem /= dims[i];
    }
}

// Превращает многомерные координаты обратно в линейный индекс для конкретного тензора (с учетом BroadCast)
inline int get_linear_index(const int* coords, __global const int* dims, int num_dims) {
    int idx = 0;
    int stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        // Если размерность равна 1, значит это бродкаст, и мы фиксируем индекс в 0 по этой оси
        int coord = (dims[i] == 1) ? 0 : coords[i];
        idx += coord * stride;
        stride *= dims[i];
    }
    return idx;
}

// Кернел для операций сложения (+) или умножения (*)
__kernel void elementwise_op(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* a_dims,
    __global const int* b_dims,
    __global const int* c_dims, // Форма выходного тензора C (максимальная по осям)
    const int num_dims,
    const int total_elements,
    const int op_type) // 0 - сложение (+), 1 - умножение (*)
{
    int tid = get_global_id(0);
    if (tid >= total_elements) return;

    int coords[10];
    
    // 1. Распаковываем индекс текущего потока в координаты выходного тензора С
    get_coords(tid, c_dims, num_dims, coords);

    // 2. Находим соответствующие линейные индексы в тензорах А и В с учетом бродкаста
    int idx_a = get_linear_index(coords, a_dims, num_dims);
    int idx_b = get_linear_index(coords, b_dims, num_dims);

    // 3. Выполняем операцию
    if (op_type == 0) {
        C[tid] = A[idx_a] + B[idx_b];
    } else {
        C[tid] = A[idx_a] * B[idx_b];
    }
}
