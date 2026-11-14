#define TILE_SIZE 16

// 1. ЯДРО ПРЯМОГО ПРОХОДА LINEAR (BMM + Broadcasting Bias в регистрах)
// X:    [Batch, M, K]
// W:    [1, K, N] (batch_stride_W = 0)
// bias: [1, N]
// Y:    [Batch, M, N]
#define TS 16          // Размер рабочей группы: 16 x 16 = 256 потоков
#define WPT 4          // Work Per Thread: каждый поток считает блок 4x4 элементов Y
#define TILE_M (TS * WPT) // 64 строки Y на группу
#define TILE_N (TS * WPT) // 64 столбца Y на группу
#define TILE_K 16      // Глубина шага по K

kernel void linear_forward_tiled(
    global const float* X,
    global const float* W,
    global const float* bias,
    global float* Y,
    const int M,
    const int K,
    const int N,
    const int batch_stride_X,
    const int batch_stride_Y
) {
    int lx = get_local_id(0);        // [0 .. 15]
    int ly = get_local_id(1);        // [0 .. 15]
    int batch_idx = get_global_id(2);

    int group_col = get_group_id(0) * TILE_N; // Начало блока 64x64 по N
    int group_row = get_group_id(1) * TILE_M; // Начало блока 64x64 по M

    global const float* current_X = X + (batch_idx * batch_stride_X);
    global float* current_Y = Y + (batch_idx * batch_stride_Y);

    // Локальная память с +1 для устранения Bank Conflicts:
    // Xsub хранит [TILE_M][TILE_K] = [64][17]
    // Wsub хранит [TILE_K][TILE_N] = [16][65]
    __local float Xsub[TILE_M][TILE_K + 1];
    __local float Wsub[TILE_K][TILE_N + 1];

    // 16 аккумуляторов в быстрых регистрах GPU (4 строки по float4)
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_K;

        // 1. Кооперативная загрузка тайла X [64 x 16] (каждый из 256 потоков грузит 4 элемента)
        // Используем branchless select() с безопасным клампингом индекса!
        int x_col = k_base + lx;
        bool valid_x_col = (x_col < K);

        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int local_r = ly + w * TS;
            int global_r = group_row + local_r;
            bool valid = (global_r < M) && valid_x_col;

            int safe_idx = select(0, global_r * K + x_col, valid);
            Xsub[local_r][lx] = select(0.0f, current_X[safe_idx], valid);
        }

        // 2. Кооперативная загрузка тайла W [16 x 64] (Coalesced чтение по N!)
        int w_row = k_base + ly;
        bool valid_w_row = (w_row < K);

        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            int local_c = lx + w * TS;
            int global_c = group_col + local_c;
            bool valid = valid_w_row && (global_c < N);

            int safe_idx = select(0, w_row * N + global_c, valid);
            Wsub[ly][local_c] = select(0.0f, W[safe_idx], valid);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Векторное умножение 4x4 в регистрах (одна инструкция на вектор!)
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Загружаем 4 элемента X по вертикали
            float4 x_vec = (float4)(
                Xsub[ly * WPT + 0][k],
                Xsub[ly * WPT + 1][k],
                Xsub[ly * WPT + 2][k],
                Xsub[ly * WPT + 3][k]
            );

            // Загружаем 4 соседних элемента W по горизонтали одним float4!
            float4 w_vec = vload4(0, &Wsub[k][lx * WPT]);

            // 4 векторных FMA-операции (16 умножений-сложений за 4 такта!)
            acc0 = fma((float4)(x_vec.x), w_vec, acc0);
            acc1 = fma((float4)(x_vec.y), w_vec, acc1);
            acc2 = fma((float4)(x_vec.z), w_vec, acc2);
            acc3 = fma((float4)(x_vec.w), w_vec, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Прибавляем векторный Bias и записываем блок 4x4 в глобальную память
    int out_col_base = group_col + lx * WPT;
    int out_row_base = group_row + ly * WPT;

    // Безопасная загрузка 4 элементов bias через select
    float4 b_vec = (float4)(
        select(0.0f, bias[select(0, out_col_base + 0, out_col_base + 0 < N)], out_col_base + 0 < N),
        select(0.0f, bias[select(0, out_col_base + 1, out_col_base + 1 < N)], out_col_base + 1 < N),
        select(0.0f, bias[select(0, out_col_base + 2, out_col_base + 2 < N)], out_col_base + 2 < N),
        select(0.0f, bias[select(0, out_col_base + 3, out_col_base + 3 < N)], out_col_base + 3 < N)
    );

    acc0 += b_vec;
    acc1 += b_vec;
    acc2 += b_vec;
    acc3 += b_vec;

    // Запись результатов (если весь блок 4x4 внутри границ — пишем одним vstore4!)
    float4 rows[4] = { acc0, acc1, acc2, acc3 };

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int gr = out_row_base + r;
        if (gr < M) {
            if (out_col_base + 3 < N) {
                vstore4(rows[r], 0, current_Y + gr * N + out_col_base);
            } else {
                // Хвост по столбцам, если N не кратно 4 (как ваше VocabSize = 6565)
                float vals[4] = { rows[r].x, rows[r].y, rows[r].z, rows[r].w };
                for (int c = 0; c < 4; c++) {
                    if (out_col_base + c < N) {
                        current_Y[gr * N + out_col_base + c] = vals[c];
                    }
                }
            }
        }
    }
}

// 2. ЯДРО ОБРАТНОГО ПРОХОДА ДЛЯ BIAS (Редукция по осям Batch и M)
// gradOutput: [Batch, M, N] -> gradBias: [1, N]
// Сетка запуска: Global = [N * 256], Local = [256] (1 рабочая группа на 1 выходной признак)
__kernel void linear_bias_backward(
    __global const float* grad_output, // [total_rows, N]
    __global float* grad_bias,         // [N]
    const int total_rows,              // Batch * M
    const int N
) {
    // Быстрая коалесцированная ось по столбцам N
    int col = get_global_id(0);
    if (col >= N) return;

    float sum = 0.0f;
    __global const float* ptr = grad_output + col;

    #pragma unroll 8
    for (int r = 0; r < total_rows; r++) {
        sum += ptr[r * N];
    }

    grad_bias[col] = sum;
}
