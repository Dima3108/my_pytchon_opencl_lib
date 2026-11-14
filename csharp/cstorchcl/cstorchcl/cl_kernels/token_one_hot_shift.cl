/**
 * Ядро для превращения буфера индексов токенов со сдвигом в One-Hot тензор для MSE Loss.
 * Сетка запуска (Global Work Size): [BATCH_SIZE, SEQ_LEN - 1]
 */
kernel void token_one_hot_shift(
    global const int* tokens,      // [BATCH_SIZE, SEQ_LEN]
    global float* destination,     // [BATCH_SIZE, SEQ_LEN - 1, VOCAB_SIZE]
    const int seq_len,
    const int vocab_size
) {
    int v = get_global_id(0);      // [0 .. vocab_size - 1] -> быстрая ось!
    int t = get_global_id(1);      // [0 .. seq_len - 2]
    int b = get_global_id(2);      // [0 .. batch_size - 1]

    int target_seq_len = seq_len - 1;
    if (v >= vocab_size || t >= target_seq_len) return;

    int token_idx = tokens[b * seq_len + (t + 1)];
    int dest_offset = (b * target_seq_len + t) * vocab_size + v;

    // Бесветвящийся выбор 1.0f на позиции истинного токена и 0.0f в остальных местах
    destination[dest_offset] = select(0.0f, 1.0f, v == token_idx);
}
