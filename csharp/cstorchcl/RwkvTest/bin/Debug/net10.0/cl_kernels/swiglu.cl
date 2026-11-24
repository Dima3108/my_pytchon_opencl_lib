#define SWI_TILE 16

// =========================================================================
// FORWARD: Тайловое умножение X * W_gate_up + SiLU(gate) * up (2.125 КБ __local)
// =========================================================================
__kernel void swiglu_gate_up_forward(
    __global const float* X,          // [M, K]
    __global const float* W_gate_up,  // [K, 2 * N]
    __global float* Out,              // [M, N]
    const int M,
    const int K,
    const int N)
{
    int lx = get_local_id(0); // [0..15]
    int ly = get_local_id(1); // [0..15]
    int col = get_group_id(0) * SWI_TILE + lx;
    int row = get_group_id(1) * SWI_TILE + ly;

    // 3 тайла по 16x17 float = 3 * 272 * 4 = 3264 байт (~3.18 КБ)
    __local float X_sub[SWI_TILE][SWI_TILE + 1];
    __local float Wg_sub[SWI_TILE][SWI_TILE + 1];
    __local float Wu_sub[SWI_TILE][SWI_TILE + 1];

    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    int two_N = 2 * N;
    int num_tiles = (K + SWI_TILE - 1) / SWI_TILE;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_k_col = t * SWI_TILE + lx;
        int tiled_k_row = t * SWI_TILE + ly;

        X_sub[ly][lx] = (row < M && tiled_k_col < K) ? X[row * K + tiled_k_col] : 0.0f;

        if (tiled_k_row < K && col < N) {
            int w_off = tiled_k_row * two_N;
            Wg_sub[ly][lx] = W_gate_up[w_off + col];
            Wu_sub[ly][lx] = W_gate_up[w_off + N + col];
        } else {
            Wg_sub[ly][lx] = 0.0f;
            Wu_sub[ly][lx] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < SWI_TILE; k++) {
            float xv = X_sub[ly][k];
            sum_gate += xv * Wg_sub[k][lx];
            sum_up   += xv * Wu_sub[k][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        float sig = 1.0f / (1.0f + exp(-sum_gate));
        Out[row * N + col] = (sum_gate * sig) * sum_up;
    }
}

// =========================================================================
// BACKWARD STEP 1: Вычисляем [d_gate | d_up] формы [M, 2*N] за один проход
// =========================================================================
__kernel void swiglu_compute_dgate_dup(
    __global const float* dOut,       // [M, N]
    __global const float* X,          // [M, K]
    __global const float* W_gate_up,  // [K, 2 * N]
    __global float* dGateUp,          // [M, 2 * N]
    const int M,
    const int K,
    const int N)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int col = get_group_id(0) * SWI_TILE + lx;
    int row = get_group_id(1) * SWI_TILE + ly;

    __local float X_sub[SWI_TILE][SWI_TILE + 1];
    __local float Wg_sub[SWI_TILE][SWI_TILE + 1];
    __local float Wu_sub[SWI_TILE][SWI_TILE + 1];

    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    int two_N = 2 * N;
    int num_tiles = (K + SWI_TILE - 1) / SWI_TILE;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_k_col = t * SWI_TILE + lx;
        int tiled_k_row = t * SWI_TILE + ly;

        X_sub[ly][lx] = (row < M && tiled_k_col < K) ? X[row * K + tiled_k_col] : 0.0f;

        if (tiled_k_row < K && col < N) {
            int w_off = tiled_k_row * two_N;
            Wg_sub[ly][lx] = W_gate_up[w_off + col];
            Wu_sub[ly][lx] = W_gate_up[w_off + N + col];
        } else {
            Wg_sub[ly][lx] = 0.0f;
            Wu_sub[ly][lx] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < SWI_TILE; k++) {
            float xv = X_sub[ly][k];
            sum_gate += xv * Wg_sub[k][lx];
            sum_up   += xv * Wu_sub[k][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        float dout_val = dOut[row * N + col];
        float sig = 1.0f / (1.0f + exp(-sum_gate));
        float silu = sum_gate * sig;
        float d_silu = sig * (1.0f + sum_gate * (1.0f - sig));

        int out_off = row * two_N;
        dGateUp[out_off + col]     = dout_val * sum_up * d_silu;
        dGateUp[out_off + N + col] = dout_val * silu;
    }
}

// =========================================================================
// BACKWARD STEP 2A: Тайловое GEMM для dX = dGateUp * W_gate_up^T (~2.1 КБ __local)
// =========================================================================
__kernel void swiglu_bwd_dx(
    __global const float* dGateUp,    // [M, 2 * N]
    __global const float* W_gate_up,  // [K, 2 * N]
    __global float* dX,               // [M, K]
    const int M,
    const int K,
    const int two_N)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int k_col = get_group_id(0) * SWI_TILE + lx; // [0 .. K-1]
    int m_row = get_group_id(1) * SWI_TILE + ly; // [0 .. M-1]

    __local float A_sub[SWI_TILE][SWI_TILE + 1];
    __local float B_sub[SWI_TILE][SWI_TILE + 1];

    float acc = 0.0f;
    int num_tiles = (two_N + SWI_TILE - 1) / SWI_TILE;

    for (int t = 0; t < num_tiles; t++) {
        int n_idx_a = t * SWI_TILE + lx;
        int n_idx_b = t * SWI_TILE + ly;

        A_sub[ly][lx] = (m_row < M && n_idx_a < two_N) ? dGateUp[m_row * two_N + n_idx_a] : 0.0f;
        // Читаем W_gate_up с транспонированием индексации: W[k_col, n_idx_b]
        B_sub[ly][lx] = (k_col < K && n_idx_b < two_N) ? W_gate_up[k_col * two_N + n_idx_b] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < SWI_TILE; i++) {
            acc += A_sub[ly][i] * B_sub[i][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m_row < M && k_col < K) {
        dX[m_row * K + k_col] = acc;
    }
}

// =========================================================================
// BACKWARD STEP 2B: Тайловое GEMM для dW = X^T * dGateUp (~2.1 КБ __local)
// =========================================================================
__kernel void swiglu_bwd_dw(
    __global const float* X,          // [M, K]
    __global const float* dGateUp,    // [M, 2 * N]
    __global float* dW_gate_up,       // [K, 2 * N]
    const int M,
    const int K,
    const int two_N)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int n_col = get_group_id(0) * SWI_TILE + lx; // [0 .. 2*N - 1]
    int k_row = get_group_id(1) * SWI_TILE + ly; // [0 .. K - 1]

    __local float X_sub[SWI_TILE][SWI_TILE + 1];
    __local float dG_sub[SWI_TILE][SWI_TILE + 1];

    float acc = 0.0f;
    int num_tiles = (M + SWI_TILE - 1) / SWI_TILE;

    for (int t = 0; t < num_tiles; t++) {
        int m_idx_x  = t * SWI_TILE + lx;
        int m_idx_dg = t * SWI_TILE + ly;

        // X^T[k_row, m_idx_x] = X[m_idx_x, k_row]
        X_sub[ly][lx]  = (k_row < K && m_idx_x < M) ? X[m_idx_x * K + k_row] : 0.0f;
        dG_sub[ly][lx] = (m_idx_dg < M && n_col < two_N) ? dGateUp[m_idx_dg * two_N + n_col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < SWI_TILE; i++) {
            acc += X_sub[ly][i] * dG_sub[i][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (k_row < K && n_col < two_N) {
        dW_gate_up[k_row * two_N + n_col] = acc;
    }
}
