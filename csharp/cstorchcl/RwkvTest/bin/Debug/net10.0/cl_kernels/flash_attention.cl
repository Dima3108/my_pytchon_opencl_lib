#define TILE_SIZE 16
#define MAX_HEAD_DIM 128

// =========================================================================
// FORWARD KERNEL: Гарантированно <= 16.2 КБ __local памяти
// =========================================================================
#define TILE_SIZE 16
#define MAX_HEAD_DIM 128

// =========================================================================
// FORWARD KERNEL: Без L_cache (0 байт дополнительного кэша в VRAM)
// =========================================================================
__kernel void flash_attention_forward(
    __global const float* Q,       // [B, N, H * D]
    __global const float* K,       // [B, N, H * D]
    __global const float* V,       // [B, N, H * D]
    __global float* O,             // [B, N, H * D]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale,
    const int is_causal)
{
    int tx = get_local_id(0);
    int q_tile_idx = get_group_id(0);
    int bh_idx = get_group_id(1);

    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    int q_row = q_tile_idx * TILE_SIZE + tx;

    int total_dim = num_heads * head_dim;
    int base_offset = b * seq_len * total_dim + h * head_dim;

    __local float K_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float V_tile[TILE_SIZE][MAX_HEAD_DIM + 1];

    float q_reg[MAX_HEAD_DIM];
    float o_acc[MAX_HEAD_DIM];
    float m_i = -1e38f;
    float l_i = 0.0f;

    if (q_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            q_reg[d] = Q[base_offset + q_row * total_dim + d];
            o_acc[d] = 0.0f;
        }
    }

    int num_kv_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        if (is_causal && kv_start > (q_tile_idx + 1) * TILE_SIZE - 1) {
            break;
        }

        int kv_row = kv_start + tx;
        for (int d = 0; d < head_dim; d++) {
            if (kv_row < seq_len) {
                K_tile[tx][d] = K[base_offset + kv_row * total_dim + d];
                V_tile[tx][d] = V[base_offset + kv_row * total_dim + d];
            } else {
                K_tile[tx][d] = 0.0f;
                V_tile[tx][d] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row < seq_len) {
            int valid_k_end = min(TILE_SIZE, seq_len - kv_start);
            for (int j = 0; j < valid_k_end; j++) {
                int k_global_pos = kv_start + j;
                if (is_causal && k_global_pos > q_row) {
                    break;
                }

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_reg[d] * K_tile[j][d];
                }
                score *= scale;

                float m_prev = m_i;
                m_i = max(m_prev, score);
                float exp_diff = exp(m_prev - m_i);
                float p_ij = exp(score - m_i);

                l_i = l_i * exp_diff + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    o_acc[d] = o_acc[d] * exp_diff + p_ij * V_tile[j][d];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row < seq_len) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            O[base_offset + q_row * total_dim + d] = o_acc[d] * inv_l;
        }
    }
}

// =========================================================================
// BACKWARD KERNEL: Без L_cache (вычисляет logsumexp на лету в регистрах)
// =========================================================================
// =========================================================================
// BACKWARD PART 1: Расчет dQ с тайлингом K и V в __local (16.125 КБ)
// =========================================================================
__kernel void flash_attention_bwd_dq(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const float* O,
    __global const float* dO,
    __global float* dQ,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale,
    const int is_causal)
{
    int tx = get_local_id(0);
    int q_tile_idx = get_group_id(0);
    int bh_idx = get_group_id(1);

    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    int q_row = q_tile_idx * TILE_SIZE + tx;

    int total_dim = num_heads * head_dim;
    int base_offset = b * seq_len * total_dim + h * head_dim;

    __local float K_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float V_tile[TILE_SIZE][MAX_HEAD_DIM + 1];

    float q_reg[MAX_HEAD_DIM];
    float do_reg[MAX_HEAD_DIM];
    float dq_acc[MAX_HEAD_DIM];
    float D_i = 0.0f;

    if (q_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            q_reg[d]  = Q[base_offset + q_row * total_dim + d];
            do_reg[d] = dO[base_offset + q_row * total_dim + d];
            float o_val = O[base_offset + q_row * total_dim + d];
            D_i += do_reg[d] * o_val;
            dq_acc[d] = 0.0f;
        }
    }

    int num_kv_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // Проход 1: Вычисляем L_i (logsumexp) через тайлы K в __local памяти
    float m_i = -1e38f;
    float l_i = 0.0f;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        if (is_causal && kv_start > (q_tile_idx + 1) * TILE_SIZE - 1) break;

        int kv_row = kv_start + tx;
        for (int d = 0; d < head_dim; d++) {
            K_tile[tx][d] = (kv_row < seq_len) ? K[base_offset + kv_row * total_dim + d] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row < seq_len) {
            int valid_k = min(TILE_SIZE, seq_len - kv_start);
            for (int j = 0; j < valid_k; j++) {
                if (is_causal && (kv_start + j) > q_row) break;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += q_reg[d] * K_tile[j][d];
                s *= scale;
                float m_prev = m_i;
                m_i = max(m_prev, s);
                l_i = l_i * exp(m_prev - m_i) + exp(s - m_i);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float L_i = m_i + log(l_i);

    // Проход 2: Накапливаем dQ через тайлы K и V в __local памяти
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        if (is_causal && kv_start > (q_tile_idx + 1) * TILE_SIZE - 1) break;

        int kv_row = kv_start + tx;
        for (int d = 0; d < head_dim; d++) {
            if (kv_row < seq_len) {
                K_tile[tx][d] = K[base_offset + kv_row * total_dim + d];
                V_tile[tx][d] = V[base_offset + kv_row * total_dim + d];
            } else {
                K_tile[tx][d] = 0.0f;
                V_tile[tx][d] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row < seq_len) {
            int valid_k = min(TILE_SIZE, seq_len - kv_start);
            for (int j = 0; j < valid_k; j++) {
                if (is_causal && (kv_start + j) > q_row) break;
                float s_ij = 0.0f;
                float dp_ij = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s_ij  += q_reg[d] * K_tile[j][d];
                    dp_ij += do_reg[d] * V_tile[j][d];
                }
                float p_ij = exp(s_ij * scale - L_i);
                float ds_ij = p_ij * (dp_ij - D_i) * scale;
                for (int d = 0; d < head_dim; d++) {
                    dq_acc[d] += ds_ij * K_tile[j][d];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            dQ[base_offset + q_row * total_dim + d] = dq_acc[d];
        }
    }
}

// =========================================================================
// BACKWARD PART 2: Расчет dK и dV с тайлингом Q и dO в __local (16.25 КБ)
// =========================================================================
__kernel void flash_attention_bwd_dkv(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const float* O,
    __global const float* dO,
    __global float* dK,
    __global float* dV,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale,
    const int is_causal)
{
    int tx = get_local_id(0);
    int kv_tile_idx = get_group_id(0);
    int bh_idx = get_group_id(1);

    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    int kv_row = kv_tile_idx * TILE_SIZE + tx;

    int total_dim = num_heads * head_dim;
    int base_offset = b * seq_len * total_dim + h * head_dim;

    // 2 * 16 * 129 * 4 + 2 * 16 * 4 = 16 640 байт (16.25 КБ < 30 КБ)
    __local float Q_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float dO_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float L_tile[TILE_SIZE];
    __local float D_tile[TILE_SIZE];

    float k_reg[MAX_HEAD_DIM];
    float v_reg[MAX_HEAD_DIM];
    float dk_acc[MAX_HEAD_DIM];
    float dv_acc[MAX_HEAD_DIM];

    if (kv_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            k_reg[d] = K[base_offset + kv_row * total_dim + d];
            v_reg[d] = V[base_offset + kv_row * total_dim + d];
            dk_acc[d] = 0.0f;
            dv_acc[d] = 0.0f;
        }
    }

    int num_q_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    int start_q_tile = is_causal ? kv_tile_idx : 0;

    for (int q_tile = start_q_tile; q_tile < num_q_tiles; q_tile++) {
        int q_start = q_tile * TILE_SIZE;
        int q_pos = q_start + tx;

        // Загружаем тайл Q, dO, вычисляем D_tile[tx] и L_tile[tx] для строки q_pos
        float d_val = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            if (q_pos < seq_len) {
                float qv = Q[base_offset + q_pos * total_dim + d];
                float dov = dO[base_offset + q_pos * total_dim + d];
                float ov = O[base_offset + q_pos * total_dim + d];
                Q_tile[tx][d] = qv;
                dO_tile[tx][d] = dov;
                d_val += dov * ov;
            } else {
                Q_tile[tx][d] = 0.0f;
                dO_tile[tx][d] = 0.0f;
            }
        }
        D_tile[tx] = d_val;

        if (q_pos < seq_len) {
            int row_max_j = is_causal ? (q_pos + 1) : seq_len;
            float m_r = -1e38f;
            float l_r = 0.0f;
            for (int jj = 0; jj < row_max_j; jj++) {
                float s_tmp = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s_tmp += Q_tile[tx][d] * K[base_offset + jj * total_dim + d];
                }
                s_tmp *= scale;
                float m_prev = m_r;
                m_r = max(m_prev, s_tmp);
                l_r = l_r * exp(m_prev - m_r) + exp(s_tmp - m_r);
            }
            L_tile[tx] = m_r + log(l_r);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (kv_row < seq_len) {
            int valid_q = min(TILE_SIZE, seq_len - q_start);
            for (int i = 0; i < valid_q; i++) {
                int q_global_pos = q_start + i;
                if (is_causal && q_global_pos < kv_row) continue;

                float s_ij = 0.0f;
                float dp_ij = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s_ij  += Q_tile[i][d] * k_reg[d];
                    dp_ij += dO_tile[i][d] * v_reg[d];
                }

                float p_ij = exp(s_ij * scale - L_tile[i]);
                float ds_ij = p_ij * (dp_ij - D_tile[i]) * scale;

                for (int d = 0; d < head_dim; d++) {
                    dv_acc[d] += p_ij * dO_tile[i][d];
                    dk_acc[d] += ds_ij * Q_tile[i][d];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (kv_row < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            dK[base_offset + kv_row * total_dim + d] = dk_acc[d];
            dV[base_offset + kv_row * total_dim + d] = dv_acc[d];
        }
    }
}