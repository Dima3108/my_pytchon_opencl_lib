#define TILE_SIZE 16

// 1. ЯДРО ПРЯМОГО ПРОХОДА LINEAR (BMM + Broadcasting Bias в регистрах)
// X:    [Batch, M, K]
// W:    [1, K, N] (batch_stride_W = 0)
// bias: [1, N]
// Y:    [Batch, M, N]
__kernel void linear_forward_tiled(
    __global const float* X,
    __global const float* W,
    __global const float* bias,
    __global float* Y,
    const int M,
    const int K,
    const int N,
    const int batch_stride_X,
    const int batch_stride_Y)
{
    int batch_idx = get_global_id(2);
    int global_col = get_global_id(0); 
    int global_row = get_global_id(1); 
    int local_col = get_local_id(0); 
    int local_row = get_local_id(1); 

    __local float Xsub[TILE_SIZE][TILE_SIZE];
    __local float Wsub[TILE_SIZE][TILE_SIZE];

    __global const float* current_X = X + (batch_idx * batch_stride_X);
    __global float* current_Y = Y + (batch_idx * batch_stride_Y);

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int x_col = t * TILE_SIZE + local_col;
        if (global_row < M && x_col < K) {
            Xsub[local_row][local_col] = current_X[global_row * K + x_col];
        } else {
            Xsub[local_row][local_col] = 0.0f;
        }

        int w_row = t * TILE_SIZE + local_row;
        if (w_row < K && global_col < N) {
            Wsub[local_row][local_col] = W[w_row * N + global_col];
        } else {
            Wsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Xsub[local_row][k] * Wsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Прибавляем Bias прямо в регистре перед записью в VRAM
    if (global_row < M && global_col < N) {
        current_Y[global_row * N + global_col] = acc + bias[global_col];
    }
}

// 2. ЯДРО ОБРАТНОГО ПРОХОДА ДЛЯ BIAS (Редукция по осям Batch и M)
// gradOutput: [Batch, M, N] -> gradBias: [1, N]
// Сетка запуска: Global = [N * 256], Local = [256] (1 рабочая группа на 1 выходной признак)
__kernel void linear_bias_backward(
    __global const float* grad_output,
    __global float* grad_bias,
    const int total_rows, // Batch * M
    const int N)
{
    int col = get_group_id(0);
    int tid = get_local_id(0);
    int gsize = get_local_size(0);

    if (col >= N) return;

    __local float l_sum[256];

    float partial_sum = 0.0f;
    for (int r = tid; r < total_rows; r += gsize) {
        partial_sum += grad_output[r * N + col];
    }
    l_sum[tid] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            l_sum[tid] += l_sum[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        grad_bias[col] = l_sum[0];
    }
}
