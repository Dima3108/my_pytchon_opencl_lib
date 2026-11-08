#define TILE_SIZE 16

// Ядро классического матричного умножения двух двумерных матриц с использованием локальной памяти.
// Матрица A имеет размерность [M x K]
// Матрица B имеет размерность [K x N]
// Выходная матрица C имеет размерность [M x N]

__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N)
{
    // Определение индексов в выходной матрице C
    int global_col = get_global_id(0); // Индекс столбца (0 до N-1)
    int global_row = get_global_id(1); // Индекс строки (0 до M-1)

    int local_col = get_local_id(0);   // Индекс потока внутри тайла по столбцам
    int local_row = get_local_id(1);   // Индекс потока внутри тайла по строкам

    // Выделение локальной памяти под тайлы матриц A и B
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    // Аккумулятор для вычисления элемента C[global_row, global_col]
    float acc = 0.0f;

    // Считаем, сколько тайлов нам нужно пройти по внутренней размерности K
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        
        // 1. Совместная загрузка элементов матрицы A в локальную память
        // Проверяем границы матрицы (защита от Out of Bounds, если размеры не кратны TILE_SIZE)
        int a_row = global_row;
        int a_col = t * TILE_SIZE + local_col;
        if (a_row < M && a_col < K) {
            Asub[local_row][local_col] = A[a_row * K + a_col];
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        // 2. Совместная загрузка элементов матрицы B в локальную память
        int b_row = t * TILE_SIZE + local_row;
        int b_col = global_col;
        if (b_row < K && b_col < N) {
            Bsub[local_row][local_col] = B[b_row * N + b_col];
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        // Ожидаем, пока ВСЕ потоки рабочей группы завершат загрузку своих элементов в локальную память
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Вычисляем частичное произведение элементов внутри текущего тайла
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Asub[local_row][k] * Bsub[k][local_col];
        }

        // Ожидаем завершения вычислений всеми потоками, перед тем как затереть локальную память на следующем шаге
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Записываем финальный результат в глобальную память C
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}
#define TILE_SIZE 16

// Ядро пакетного умножения матриц (Batch Matrix Multiplication)
// А имеет форму [Batch, M, K]
// В имеет форму [Batch, K, N]
// С имеет форму [Batch, M, N]
__kernel void bmm_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N,
    const int batch_stride_A, // M * K
    const int batch_stride_B, // K * N
    const int batch_stride_C) // M * N
{
    int batch_idx = get_global_id(2); // Индекс матрицы в пакете (Batch)
    int global_col = get_global_id(0); 
    int global_row = get_global_id(1); 

    int local_col = get_local_id(0);   
    int local_row = get_local_id(1);   

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    // Смещаем указатели глобальной памяти под текущий батч
    __global const float* current_A = A + (batch_idx * batch_stride_A);
    __global const float* current_B = B + (batch_idx * batch_stride_B);
    __global float* current_C = C + (batch_idx * batch_stride_C);

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + local_col;
        if (global_row < M && a_col < K) {
            Asub[local_row][local_col] = current_A[global_row * K + a_col];
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        int b_row = t * TILE_SIZE + local_row;
        if (b_row < K && global_col < N) {
            Bsub[local_row][local_col] = current_B[b_row * N + global_col];
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
