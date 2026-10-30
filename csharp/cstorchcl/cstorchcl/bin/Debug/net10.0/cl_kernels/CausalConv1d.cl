// Входной тензор имеет форму [Batch, Channels, Length] (после вашего transpose)
// Веса имеют форму [Channels, Kernel_Size] (так как свертка depthwise)
// Bias имеет форму [Channels]

__kernel void causal_conv1d_depthwise(
    __global const float* X,      // [Batch, Channels, Length]
    __global const float* weight, // [Channels, Kernel_Size]
    __global const float* bias,   // [Channels]
    __global float* Y,            // [Batch, Channels, Length]
    const int batch_size,
    const int channels,
    const int length,
    const int kernel_size)
{
    // Каждый поток обрабатывает одну точку на временной оси для конкретного канала конкретного батча
    int t = get_global_id(0);       // Индекс по времени (0 ... length-1)
    int c = get_global_id(1);       // Индекс канала (0 ... channels-1)
    int b = get_global_id(2);       // Индекс батча (0 ... batch_size-1)

    if (t >= length || c >= channels || b >= batch_size) return;

    // Базовое смещение для текущего канала в текущем батче
    int channel_stride = length;
    int batch_stride = channels * length;
    int x_base_idx = b * batch_stride + c * channel_stride;

    float sum = bias[c];

    // Выполняем свертку назад по времени (каузально)
    for (int k = 0; k < kernel_size; k++) {
        // Текущий индекс во времени с учетом сдвига ядра свертки
        int input_t = t - (kernel_size - 1 - k);

        float x_val = 0.0f;
        // Каузальный паддинг: если индекс валидный — берем значение, если меньше нуля — оставляем 0.0f
        if (input_t >= 0) {
            x_val = X[x_base_idx + input_t];
        }

        // Индекс веса для текущего канала c и текущей позиции ядра k
        int w_idx = c * kernel_size + k;
        sum += x_val * weight[w_idx];
    }

    // Записываем результат в выходной тензор Y
    int y_idx = b * batch_stride + c * channel_stride + t;
    Y[y_idx] = sum;
}

// Ядро 1: Вычисление градиента по входу (grad_X). 
// Выполняет зеркальный (антикаузальный) проход по времени.
__kernel void causal_conv1d_backward_input(
    __global const float* grad_output, // [Batch, Channels, Length]
    __global const float* weight,      // [Channels, Kernel_Size]
    __global float* grad_X,            // [Batch, Channels, Length]
    const int batch_size,
    const int channels,
    const int length,
    const int kernel_size)
{
    int t = get_global_id(0); // Индекс по времени (0 ... length-1)
    int c = get_global_id(1); // Индекс канала (0 ... channels-1)
    int b = get_global_id(2); // Индекс батча (0 ... batch_size-1)

    if (t >= length || c >= channels || b >= batch_size) return;

    int channel_stride = length;
    int batch_stride = channels * length;
    int base_idx = b * batch_stride + c * channel_stride;

    float sum_grad_x = 0.0f;

    // Антикаузальный проход: собираем градиенты из текущего и будущего времени
    for (int k = 0; k < kernel_size; k++) {
        // Вычисляем, в какую точку будущего улетел вклад текущего элемента
        int output_t = t + (kernel_size - 1 - k);

        // Проверяем границы: если индекс времени не вышел за пределы Length справа
        if (output_t < length) {
            int go_idx = base_idx + output_t;
            int w_idx = c * kernel_size + k;
            sum_grad_x += grad_output[go_idx] * weight[w_idx];
        }
    }

    grad_X[base_idx + t] = sum_grad_x;
}

// Ядро 2: Вычисление градиента по весам (grad_weight) и bias.
// Каждый поток отвечает за конкретную ячейку фильтра конкретного канала.
// Потоки суммируют вклады по всей оси времени и по всему батчу.
__kernel void causal_conv1d_backward_weights(
    __global const float* grad_output, // [Batch, Channels, Length]
    __global const float* X,           // [Batch, Channels, Length]
    __global float* grad_weight,       // [Channels, Kernel_Size]
    __global float* grad_bias,         // [Channels]
    const int batch_size,
    const int channels,
    const int length,
    const int kernel_size)
{
    int k = get_global_id(0); // Позиция в ядре свертки (0 ... kernel_size-1)
    int c = get_global_id(1); // Канал (0 ... channels-1)

    if (k >= kernel_size || c >= channels) return;

    int channel_stride = length;
    int batch_stride = channels * length;

    float sum_weight = 0.0f;
    float sum_bias = 0.0f;

    // Суммируем градиенты по временной оси и по всему пакету (Batch)
    
    for (int b = 0; b < batch_size; b++) {
        int base_idx = b * batch_stride + c * channel_stride;
        #pragma unroll
        for (int t = 0; t < length; t++) {
            int go_idx = base_idx + t;
            
            // Считаем градиент весов с учетом виртуального каузального паддинга прямого прохода
            int input_t = t - (kernel_size - 1 - k);
            float x_val = 0.0f;
            if (input_t >= 0) {
                x_val = X[base_idx + input_t];
            }

            sum_weight += grad_output[go_idx] * x_val;
            
            
            // Градиент по bias считается только один раз (например, потоком k == 0)
            if (k == 0) {
                sum_bias += grad_output[go_idx];
            }
        }
    }

    // Записываем результат
    grad_weight[c * kernel_size + k] = sum_weight;
    if (k == 0) {
        grad_bias[c] = sum_bias;
    }
}
