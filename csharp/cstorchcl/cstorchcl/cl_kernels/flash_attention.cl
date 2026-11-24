#define TILE_SIZE 16
#define MAX_HEAD_DIM 128
#define LN_10000_2 -18.420680743952367f // -2.0 * ln(10000.0)

// =========================================================================
// БЫСТРЫЙ ROPE НА FLOAT2 БЕЗ POW() (Геометрическая прогрессия частот)
// =========================================================================
inline void apply_rope_float2(float* vec, const int pos, const int head_dim, const float dir)
{
    int half_dim = head_dim >> 1;
    float f_pos = (float)pos * dir;
    // Один exp() вместо half_dim вызовов pow()!
    float freq_step = exp(LN_10000_2 / (float)head_dim);
    float theta = 1.0f;

    for (int i = 0; i < half_dim; i++) {
        float sin_val;
        float cos_val = sincos(f_pos * theta, &sin_val);

        int idx = i << 1;
        float2 v = (float2)(vec[idx], vec[idx + 1]);
        float2 rot = (float2)(
            v.x * cos_val - v.y * sin_val,
            v.x * sin_val + v.y * cos_val
        );

        vec[idx]     = rot.x;
        vec[idx + 1] = rot.y;

        theta *= freq_step;
    }
}

// =========================================================================
// 1. FORWARD KERNEL: Branchless + Fast RoPE
// =========================================================================
__kernel void flash_attention_forward(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* O,
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
    int active_q = (q_row < seq_len);

    int total_dim = num_heads * head_dim;
    int base_offset = b * seq_len * total_dim + h * head_dim;

    __local float K_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float V_tile[TILE_SIZE][MAX_HEAD_DIM + 1];

    float q_reg[MAX_HEAD_DIM];
    float o_acc[MAX_HEAD_DIM];
    float m_i = -1e38f;
    float l_i = 0.0f;

    int safe_q_row = min(q_row, seq_len - 1);
    for (int d = 0; d < head_dim; d++) {
        q_reg[d] = active_q ? Q[base_offset + safe_q_row * total_dim + d] : 0.0f;
        o_acc[d] = 0.0f;
    }
    if (active_q) {
        apply_rope_float2(q_reg, q_row, head_dim, 1.0f);
    }

    int total_kv_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    int max_q_in_tile = min((q_tile_idx + 1) * TILE_SIZE, seq_len);
    int loop_kv_tiles = is_causal ? ((max_q_in_tile + TILE_SIZE - 1) / TILE_SIZE) : total_kv_tiles;

    for (int kv_tile = 0; kv_tile < loop_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        int kv_row = kv_start + tx;
        int valid_kv = (kv_row < seq_len);
        int safe_kv_row = min(kv_row, seq_len - 1);

        float k_tmp[MAX_HEAD_DIM];
        for (int d = 0; d < head_dim; d++) {
            k_tmp[d]      = valid_kv ? K[base_offset + safe_kv_row * total_dim + d] : 0.0f;
            V_tile[tx][d] = valid_kv ? V[base_offset + safe_kv_row * total_dim + d] : 0.0f;
        }
        if (valid_kv) {
            apply_rope_float2(k_tmp, kv_row, head_dim, 1.0f);
        }
        for (int d = 0; d < head_dim; d++) {
            K_tile[tx][d] = k_tmp[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Вычисляем точную границу j без ветвлений внутри цикла
        int valid_k = min(TILE_SIZE, seq_len - kv_start);
        int j_end = active_q ? (is_causal ? clamp(q_row - kv_start + 1, 0, valid_k) : valid_k) : 0;

        for (int j = 0; j < j_end; j++) {
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (active_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            O[base_offset + q_row * total_dim + d] = o_acc[d] * inv_l;
        }
    }
}

// =========================================================================
// 2. BACKWARD PART 1: Расчет dQ (Без ветвлений в циклах + Fast RoPE)
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
    int active_q = (q_row < seq_len);

    int total_dim = num_heads * head_dim;
    int base_offset = b * seq_len * total_dim + h * head_dim;

    __local float K_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
    __local float V_tile[TILE_SIZE][MAX_HEAD_DIM + 1];

    float q_reg[MAX_HEAD_DIM];
    float do_reg[MAX_HEAD_DIM];
    float dq_acc[MAX_HEAD_DIM];
    float D_i = 0.0f;

    int safe_q_row = min(q_row, seq_len - 1);
    for (int d = 0; d < head_dim; d++) {
        q_reg[d]  = active_q ? Q[base_offset + safe_q_row * total_dim + d] : 0.0f;
        do_reg[d] = active_q ? dO[base_offset + safe_q_row * total_dim + d] : 0.0f;
        float o_v = active_q ? O[base_offset + safe_q_row * total_dim + d] : 0.0f;
        D_i += do_reg[d] * o_v;
        dq_acc[d] = 0.0f;
    }
    if (active_q) {
        apply_rope_float2(q_reg, q_row, head_dim, 1.0f);
    }

    int total_kv_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    int max_q_in_tile = min((q_tile_idx + 1) * TILE_SIZE, seq_len);
    int loop_kv_tiles = is_causal ? ((max_q_in_tile + TILE_SIZE - 1) / TILE_SIZE) : total_kv_tiles;

    // Проход 1: Вычисляем L_i (logsumexp) без ветвлений в цикле
    float m_i = -1e38f;
    float l_i = 0.0f;

    for (int kv_tile = 0; kv_tile < loop_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        int kv_row = kv_start + tx;
        int valid_kv = (kv_row < seq_len);
        int safe_kv_row = min(kv_row, seq_len - 1);

        float k_tmp[MAX_HEAD_DIM];
        for (int d = 0; d < head_dim; d++) {
            k_tmp[d] = valid_kv ? K[base_offset + safe_kv_row * total_dim + d] : 0.0f;
        }
        if (valid_kv) {
            apply_rope_float2(k_tmp, kv_row, head_dim, 1.0f);
        }
        for (int d = 0; d < head_dim; d++) {
            K_tile[tx][d] = k_tmp[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int valid_k = min(TILE_SIZE, seq_len - kv_start);
        int j_end = active_q ? (is_causal ? clamp(q_row - kv_start + 1, 0, valid_k) : valid_k) : 0;

        for (int j = 0; j < j_end; j++) {
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s += q_reg[d] * K_tile[j][d];
            }
            s *= scale;
            float m_prev = m_i;
            m_i = max(m_prev, s);
            l_i = l_i * exp(m_prev - m_i) + exp(s - m_i);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float L_i = m_i + log(l_i);

    // Проход 2: Накапливаем градиент dQ без ветвлений в цикле
    for (int kv_tile = 0; kv_tile < loop_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        int kv_row = kv_start + tx;
        int valid_kv = (kv_row < seq_len);
        int safe_kv_row = min(kv_row, seq_len - 1);

        float k_tmp[MAX_HEAD_DIM];
        for (int d = 0; d < head_dim; d++) {
            k_tmp[d]      = valid_kv ? K[base_offset + safe_kv_row * total_dim + d] : 0.0f;
            V_tile[tx][d] = valid_kv ? V[base_offset + safe_kv_row * total_dim + d] : 0.0f;
        }
        if (valid_kv) {
            apply_rope_float2(k_tmp, kv_row, head_dim, 1.0f);
        }
        for (int d = 0; d < head_dim; d++) {
            K_tile[tx][d] = k_tmp[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int valid_k = min(TILE_SIZE, seq_len - kv_start);
        int j_end = active_q ? (is_causal ? clamp(q_row - kv_start + 1, 0, valid_k) : valid_k) : 0;

        for (int j = 0; j < j_end; j++) {
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (active_q) {
        apply_rope_float2(dq_acc, q_row, head_dim, -1.0f);
        for (int d = 0; d < head_dim; d++) {
            dQ[base_offset + q_row * total_dim + d] = dq_acc[d];
        }
    }
}

// =========================================================================
// 3. BACKWARD PART 2: Расчет dK и dV (Fast RoPE + тайлинг K при расчете L)
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
int active_kv = (kv_row < seq_len);
int total_dim = num_heads * head_dim;
int base_offset = b * seq_len * total_dim + h * head_dim;
__local float Q_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
__local float dO_tile[TILE_SIZE][MAX_HEAD_DIM + 1];
__local float L_tile[TILE_SIZE];
__local float D_tile[TILE_SIZE];
float k_reg[MAX_HEAD_DIM];
float v_reg[MAX_HEAD_DIM];
float dk_acc[MAX_HEAD_DIM];
float dv_acc[MAX_HEAD_DIM];
int safe_kv_row = min(kv_row, seq_len - 1);
for (int d = 0; d < head_dim; d++) {
k_reg[d] = active_kv ? K[base_offset + safe_kv_row * total_dim + d] : 0.0f;
v_reg[d] = active_kv ? V[base_offset + safe_kv_row * total_dim + d] : 0.0f;
dk_acc[d] = 0.0f;
dv_acc[d] = 0.0f;
}
if (active_kv) {
apply_rope_float2(k_reg, kv_row, head_dim, 1.0f);
}
int num_q_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
int start_q_tile = is_causal ? kv_tile_idx : 0;
for (int q_tile = start_q_tile; q_tile < num_q_tiles; q_tile++) {
int q_start = q_tile * TILE_SIZE;
int q_pos = q_start + tx;
int valid_q_pos = (q_pos < seq_len);
int safe_q_pos = min(q_pos, seq_len - 1);
float d_val = 0.0f;
float q_tmp[MAX_HEAD_DIM];
for (int d = 0; d < head_dim; d++) {
q_tmp[d]  = valid_q_pos ? Q[base_offset + safe_q_pos * total_dim + d] : 0.0f;
float dov = valid_q_pos ? dO[base_offset + safe_q_pos * total_dim + d] : 0.0f;
float ov  = valid_q_pos ? O[base_offset + safe_q_pos * total_dim + d] : 0.0f;
dO_tile[tx][d] = dov;
d_val += dov * ov;
}
if (valid_q_pos) {
apply_rope_float2(q_tmp, q_pos, head_dim, 1.0f);
}
for (int d = 0; d < head_dim; d++) {
Q_tile[tx][d] = q_tmp[d];
}
D_tile[tx] = d_val;
if (valid_q_pos) {
int row_max_j = is_causal ? (q_pos + 1) : seq_len;
float m_r = -1e38f;
float l_r = 0.0f;
for (int jj = 0; jj < row_max_j; jj++) {
float k_jj[MAX_HEAD_DIM];
for (int d = 0; d < head_dim; d++) {
k_jj[d] = K[base_offset + jj * total_dim + d];
}
apply_rope_float2(k_jj, jj, head_dim, 1.0f);
float s_tmp = 0.0f;
for (int d = 0; d < head_dim; d++) {
s_tmp += Q_tile[tx][d] * k_jj[d];
}
s_tmp *= scale;
float m_prev = m_r;
m_r = max(m_prev, s_tmp);
l_r = l_r * exp(m_prev - m_r) + exp(s_tmp - m_r);
}
L_tile[tx] = m_r + log(l_r);
} else {
L_tile[tx] = 0.0f;
}
barrier(CLK_LOCAL_MEM_FENCE);
int valid_q = min(TILE_SIZE, seq_len - q_start);
int i_start = active_kv ? (is_causal ? clamp(kv_row - q_start, 0, valid_q) : 0) : valid_q;
for (int i = i_start; i < valid_q; i++) {
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
barrier(CLK_LOCAL_MEM_FENCE);
}
if (active_kv) {
apply_rope_float2(dk_acc, kv_row, head_dim, -1.0f);
for (int d = 0; d < head_dim; d++) {
dK[base_offset + kv_row * total_dim + d] = dk_acc[d];
dV[base_offset + kv_row * total_dim + d] = dv_acc[d];
}
}
}