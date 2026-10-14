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
