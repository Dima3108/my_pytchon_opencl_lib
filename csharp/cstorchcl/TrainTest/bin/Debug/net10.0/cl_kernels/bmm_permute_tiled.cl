#define TILE_SIZE 16

__kernel void bmm_permute_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N,
    const int batch_stride_A,
    const int batch_stride_B,
    const int batch_stride_C,
    const int transA, // 1 если A нужно транспонировать [Batch, K, M] -> [Batch, M, K]
    const int transB) // 1 если B нужно транспонировать [Batch, N, K] -> [Batch, K, N]
{
    int batch_idx = get_global_id(2);
    int global_col = get_global_id(0);
    int global_row = get_global_id(1);
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    __global const float* current_A = A + (batch_idx * batch_stride_A);
    __global const float* current_B = B + (batch_idx * batch_stride_B);
    __global float* current_C = C + (batch_idx * batch_stride_C);

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Загрузка тайла из A (учитываем transA)
        int a_col = t * TILE_SIZE + local_col;
        if (global_row < M && a_col < K) {
            Asub[local_row][local_col] = (transA == 0)
                ? current_A[global_row * K + a_col]   // Обычное чтение [M, K]
                : current_A[a_col * M + global_row];  // Транспонированное [K, M]
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        // Загрузка тайла из B (учитываем transB)
        int b_row = t * TILE_SIZE + local_row;
        if (b_row < K && global_col < N) {
            Bsub[local_row][local_col] = (transB == 0)
                ? current_B[b_row * N + global_col]   // Обычное чтение [K, N]
                : current_B[global_col * K + b_row];  // Транспонированное [N, K]
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        current_C[global_row * N + global_col] = acc;
    }
}
