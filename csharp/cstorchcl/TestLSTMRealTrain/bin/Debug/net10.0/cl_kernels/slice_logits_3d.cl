kernel void slice_logits_3d(
    global const float* src,       // [Batch, OldSeqLen, VocabSize]
    global float* dst,             // [Batch, NewSeqLen, VocabSize]
    const int old_seq_len,
    const int new_seq_len,
    const int vocab_size
) {
    int v_base = get_global_id(0) * 4; // Векторный шаг по 4 элемента
    int t = get_global_id(1);          // [0 .. new_seq_len - 1]
    int b = get_global_id(2);          // [0 .. batch_size - 1]

    if (v_base >= vocab_size || t >= new_seq_len) return;

    int src_offset = (b * old_seq_len + t) * vocab_size + v_base;
    int dst_offset = (b * new_seq_len + t) * vocab_size + v_base;

    if (v_base + 3 < vocab_size) {
        vstore4(vload4(0, src + src_offset), 0, dst + dst_offset);
    } else {
        for (int i = 0; i < 4 && (v_base + i) < vocab_size; i++) {
            dst[dst_offset + i] = src[src_offset + i];
        }
    }
}
