/**
 * Ядро для превращения буфера индексов токенов со сдвигом в One-Hot тензор для MSE Loss.
 * Сетка запуска (Global Work Size): [BATCH_SIZE, SEQ_LEN - 1]
 */
kernel void token_one_hot_shift(
    global const int* tokens,          // Исходный буфер токенов [BATCH_SIZE, SEQ_LEN]
    global float* destination,         // Выходной тензор мишени [BATCH_SIZE, SEQ_LEN - 1, VOCAB_SIZE]
    const int seq_len,
    const int vocab_size
) {
    int b = get_global_id(0); // Индекс батча
    int t = get_global_id(1); // Индекс шага времени (уже в диапазоне от 0 до SEQ_LEN - 2)

    // Применяем авторегрессионный сдвиг: берем токен из позиции t + 1 (эквивалент targets[:, 1:])
    int token_idx = tokens[b * seq_len + (t + 1)];

    // Вычисляем базовое смещение в выходном 3D тензоре
    int dest_row_offset = b * ((seq_len - 1) * vocab_size) + t * vocab_size;

    // Заполняем строку вектора: на месте истинного токена ставим 1.0f, в остальных местах 0.0f
    for (int v = 0; v < vocab_size; v++) {
        destination[dest_row_offset + v] = (v == token_idx) ? 1.0f : 0.0f;
    }
}
