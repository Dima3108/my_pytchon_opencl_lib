#define TS 16
#define WPT 4
#define TILE_M (TS * WPT) // 64
#define TILE_N (TS * WPT) // 64
#define TILE_K 16

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
    const int transB  // 1 если B нужно транспонировать [Batch, N, K] -> [Batch, K, N]
) {
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int batch_idx = get_global_id(2);

    int group_col = get_group_id(0) * TILE_N; // Начало блока 64x64 по N
    int group_row = get_group_id(1) * TILE_M; // Начало блока 64x64 по M

    __global const float* current_A = A + (batch_idx * batch_stride_A);
    __global const float* current_B = B + (batch_idx * batch_stride_B);
    __global float* current_C       = C + (batch_idx * batch_stride_C);

    // +1 устраняет Bank Conflicts в локальной памяти
    __local float Asub[TILE_M][TILE_K + 1];
    __local float Bsub[TILE_K][TILE_N + 1];

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    // Инвариантные флаги транспонирования вынесены за цикл
    int is_trans_A = (transA != 0);
    int is_trans_B = (transB != 0);

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_K;

        // 1. Кооперативная бесветвящаяся загрузка тайла A [64 x 16]
        int a_col = k_base + lx;
        int valid_a_col = (a_col < K);

        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int local_r = ly + w * TS;
            int global_r = group_row + local_r;
            int valid_a = (global_r < M) && valid_a_col;

            int idx_a = select(global_r * K + a_col, a_col * M + global_r, is_trans_A);
            int safe_a = select(0, idx_a, valid_a);
            Asub[local_r][lx] = select(0.0f, current_A[safe_a], valid_a);
        }

        // 2. Кооперативная бесветвящаяся загрузка тайла B [16 x 64]
        int b_row = k_base + ly;
        int valid_b_row = (b_row < K);

        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int local_c = lx + w * TS;
            int global_c = group_col + local_c;
            int valid_b = valid_b_row && (global_c < N);

            int idx_b = select(b_row * N + global_c, global_c * K + b_row, is_trans_B);
            int safe_b = select(0, idx_b, valid_b);
            Bsub[ly][local_c] = select(0.0f, current_B[safe_b], valid_b);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Векторное 4x4 умножение в регистрах через FMA
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float4 a_vec = (float4)(
                Asub[ly * WPT + 0][k],
                Asub[ly * WPT + 1][k],
                Asub[ly * WPT + 2][k],
                Asub[ly * WPT + 3][k]
            );
            float4 b_vec = vload4(0, &Bsub[k][lx * WPT]);

            acc0 = fma((float4)(a_vec.x), b_vec, acc0);
            acc1 = fma((float4)(a_vec.y), b_vec, acc1);
            acc2 = fma((float4)(a_vec.z), b_vec, acc2);
            acc3 = fma((float4)(a_vec.w), b_vec, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Векторная запись блока 4x4 в выходную матрицу C
    int out_col_base = group_col + lx * WPT;
    int out_row_base = group_row + ly * WPT;
    float4 rows[4] = { acc0, acc1, acc2, acc3 };

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int gr = out_row_base + r;
        if (gr < M) {
            if (out_col_base + 3 < N) {
                vstore4(rows[r], 0, current_C + gr * N + out_col_base);
            } else {
                float vals[4] = { rows[r].x, rows[r].y, rows[r].z, rows[r].w };
                for (int c = 0; c < 4; c++) {
                    if (out_col_base + c < N) {
                        current_C[gr * N + out_col_base + c] = vals[c];
                    }
                }
            }
        }
    }
}
