__kernel void tensor_cat(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* a_dims,
    __global const int* b_dims,
    __global const int* c_dims,
    const int num_dims,
    const int cat_axis,
    const int total_elements) 
{
    int tid = get_global_id(0);
    if (tid >= total_elements) return;

    int coords[10];
    get_coords(tid, c_dims, num_dims, coords);

    // Граница, где заканчивается тензор A и начинается тензор B по оси конкатенации
    int boundary = a_dims[cat_axis];

    if (coords[cat_axis] < boundary) {
        // Данные берутся из тензора A
        int idx_a = 0;
        int stride = 1;
        for (int i = num_dims - 1; i >= 0; i--) {
            idx_a += coords[i] * stride;
            stride *= a_dims[i];
        }
        C[tid] = A[idx_a];
    } else {
        // Данные берутся из тензора B (сдвигаем координату назад на размер A)
        int coords_b[10];
        for(int i = 0; i < num_dims; i++) coords_b[i] = coords[i];
        coords_b[cat_axis] -= boundary;

        int idx_b = 0;
        int stride = 1;
        for (int i = num_dims - 1; i >= 0; i--) {
            idx_b += coords_b[i] * stride;
            stride *= b_dims[i];
        }
        C[tid] = B[idx_b];
    }
}
