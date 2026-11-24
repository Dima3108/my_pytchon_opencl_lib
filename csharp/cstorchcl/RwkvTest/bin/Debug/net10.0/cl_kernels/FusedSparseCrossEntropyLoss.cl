#define CE_WG_SIZE 256


kernel void sparse_cross_entropy_forward(
    global const float* logits,       // [Batch, SeqLen, VocabSize]
    global const int* targets,        // [Batch, SeqLen]
    global float* token_losses,       // [Batch * (SeqLen - 1)]
    const int seq_len,
    const int vocab_size
) {
    int lid = get_local_id(0);
    int t   = get_group_id(1);
    int b   = get_group_id(2);

    int target_seq_len = seq_len - 1;
    if (t >= target_seq_len) return;

    __local float shared_buf[CE_WG_SIZE];

    int target_id = targets[b * seq_len + (t + 1)];
    global const float* row_logits = logits + (b * seq_len + t) * vocab_size;

    // 1. Поиск максимума по словарю
    float local_max = -INFINITY;
    for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
        local_max = fmax(local_max, row_logits[v]);
    }
    shared_buf[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CE_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_buf[lid] = fmax(shared_buf[lid], shared_buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = shared_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Сумма экспонент
    float local_sum = 0.0f;
    for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
        local_sum += native_exp(row_logits[v] - max_val);
    }
    shared_buf[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CE_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_buf[lid] += shared_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float sum_exp = shared_buf[0];
        float target_logit = (target_id >= 0 && target_id < vocab_size) ? row_logits[target_id] : max_val;
        float loss = log(sum_exp) - (target_logit - max_val);
        token_losses[b * target_seq_len + t] = loss;
    }
}

kernel void reduce_mean_loss(
    global const float* token_losses,
    global float* out_loss,
    const int num_tokens
) {
    int lid = get_local_id(0);
    __local float shared_sum[CE_WG_SIZE];

    float local_sum = 0.0f;
    for (int i = lid; i < num_tokens; i += CE_WG_SIZE) {
        local_sum += token_losses[i];
    }
    shared_sum[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CE_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_loss[0] = shared_sum[0] / (float)num_tokens;
    }
}

kernel void sparse_cross_entropy_backward(
    global const float* logits,
    global const int* targets,
    global const float* grad_loss,
    global float* grad_logits,
    const int batch_size,
    const int seq_len,
    const int vocab_size
) {
    int lid = get_local_id(0);
    int t   = get_group_id(1);
    int b   = get_group_id(2);

    int target_seq_len = seq_len - 1;
    global float* row_grad = grad_logits + (b * seq_len + t) * vocab_size;

    if (t >= target_seq_len) {
        for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
            row_grad[v] = 0.0f;
        }
        return;
    }

    __local float shared_buf[CE_WG_SIZE];

    int target_id = targets[b * seq_len + (t + 1)];
    global const float* row_logits = logits + (b * seq_len + t) * vocab_size;

    // 1. Поиск максимума по словарю
    float local_max = -INFINITY;
    for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
        local_max = fmax(local_max, row_logits[v]);
    }
    shared_buf[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CE_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_buf[lid] = fmax(shared_buf[lid], shared_buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = shared_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Сумма экспонент
    float local_sum = 0.0f;
    for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
        local_sum += native_exp(row_logits[v] - max_val);
    }
    shared_buf[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CE_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_buf[lid] += shared_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_sum_exp = 1.0f / shared_buf[0];
    float scale = grad_loss[0] / (float)(batch_size * target_seq_len);

    for (int v = lid; v < vocab_size; v += CE_WG_SIZE) {
        float prob = native_exp(row_logits[v] - max_val) * inv_sum_exp;
        float indicator = select(0.0f, 1.0f, v == target_id);
        row_grad[v] = (prob - indicator) * scale;
    }
}
