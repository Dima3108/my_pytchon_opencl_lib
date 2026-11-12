// Полезные инлайн-функции активации
inline float sigmoid(float x) { return 1.0f / (1.0f + native_exp(-x)); }

inline float4 sigmoid4(float4 x) { return 1.0f / (1.0f + native_exp(-x)); }

void atomic_add_float(global volatile float *source, const float val) {
  if (val == 0.0f)
    return;
  union {
    unsigned int intVal;
    float floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal + val;
    current.intVal = atomic_cmpxchg((global volatile unsigned int *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}

// ============================================================================
// ПРЯМОЙ ПРОХОД (FORWARD) С ВЕКТОРИЗАЦИЕЙ АКТИВАЦИЙ ПО 4 НЕЙРОНА
// ============================================================================
#define TILE_SIZE 16
#define TS 16
#define WPT 4
#define TILE_M (TS * WPT) // 64
#define TILE_N (TS * WPT) // 64
#define TILE_K 16

// ============================================================================
// 1. ПРЯМОЙ ПРОХОД GEMM: XW_out = X * W + bias (4x4 Register Tiling)
// ============================================================================

kernel void lstm_recurrent_forward(
    global const float
        *XW_in, // Готовая проекция X*W + bias: [Batch, SeqLen, 4 * HiddenSize]
    global const float *U,    // [HiddenSize, 4 * HiddenSize]
    global float *H_out,      // [Batch, SeqLen, HiddenSize]
    global float *workspace,  // [SeqLen, Batch, 5 * HiddenSize]
    global float *c_curr_buf, // [Batch, HiddenSize]
    global float *h_curr_buf, // [Batch, HiddenSize]
    const int batch_size, const int seq_len, const int hidden_size) {
  int b = get_group_id(0);
  int lid = get_local_id(1);
  int wg_size = get_local_size(1);

  if (b >= batch_size)
    return;

  global float *c_curr_ptr = c_curr_buf + (b * hidden_size);
  global float *h_curr_ptr = h_curr_buf + (b * hidden_size);

  for (int hid = lid; hid < hidden_size; hid += wg_size) {
    c_curr_ptr[hid] = 0.0f;
    h_curr_ptr[hid] = 0.0f;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  int total_gates = 4 * hidden_size;
  int ws_stride = 5 * hidden_size;
  int vec_hidden = (hidden_size / 4) * 4;

  for (int t = 0; t < seq_len; t++) {
    // XW_in имеет форму [Batch, SeqLen, 4 * HiddenSize]
    int xw_offset = (b * seq_len + t) * total_gates;
    int ws_offset = (t * batch_size + b) * ws_stride;
    int h_out_offset = (b * seq_len + t) * hidden_size;

    // 1. Только рекуррентное умножение H_{t-1} * U + готовый XW_in
    for (int hid = lid; hid < hidden_size; hid += wg_size) {
      for (int g = 0; g < 4; g++) {
        int gate_idx = g * hidden_size + hid;
        float val = XW_in[xw_offset + gate_idx];

        int h = 0;
        for (; h <= hidden_size - 4; h += 4) {
          float4 h_vec = vload4(0, h_curr_ptr + h);
          float4 u_vec = (float4)(U[(h + 0) * total_gates + gate_idx],
                                  U[(h + 1) * total_gates + gate_idx],
                                  U[(h + 2) * total_gates + gate_idx],
                                  U[(h + 3) * total_gates + gate_idx]);
          val += dot(h_vec, u_vec);
        }
        for (; h < hidden_size; h++) {
          val += h_curr_ptr[h] * U[h * total_gates + gate_idx];
        }

        workspace[ws_offset + gate_idx] = val;
      }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // 2. Векторизованные активации по 4 нейрона за итерацию
    for (int hid = lid * 4; hid < vec_hidden; hid += wg_size * 4) {
      float4 i_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 0 * hidden_size + hid));
      float4 f_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 1 * hidden_size + hid));
      float4 g_gate =
          tanh(vload4(0, workspace + ws_offset + 2 * hidden_size + hid));
      float4 o_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 3 * hidden_size + hid));

      float4 c_prev = vload4(0, c_curr_ptr + hid);
      float4 c_val = f_gate * c_prev + i_gate * g_gate;

      vstore4(c_val, 0, c_curr_ptr + hid);
      vstore4(c_val, 0, workspace + ws_offset + 4 * hidden_size + hid);

      float4 h_val = o_gate * tanh(c_val);
      vstore4(h_val, 0, h_curr_ptr + hid);
      vstore4(h_val, 0, H_out + h_out_offset + hid);
    }

    // Хвост, если hidden_size не кратен 4
    for (int hid = vec_hidden + lid; hid < hidden_size; hid += wg_size) {
      float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + hid]);
      float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + hid]);
      float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + hid]);
      float o_gate = sigmoid(workspace[ws_offset + 3 * hidden_size + hid]);

      float c_val = f_gate * c_curr_ptr[hid] + i_gate * g_gate;
      c_curr_ptr[hid] = c_val;
      workspace[ws_offset + 4 * hidden_size + hid] = c_val;

      float h_val = o_gate * tanh(c_val);
      h_curr_ptr[hid] = h_val;
      H_out[h_out_offset + hid] = h_val;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

// ============================================================================
// ОБРАТНЫЙ ПРОХОД (BACKWARD) БЕЗ АТОМИКОВ ВНУТРИ ЦИКЛА ПО ВРЕМЕНИ
// ============================================================================
// ============================================================================
// ЯДРО 1: ЧИСТЫЙ РЕКУРРЕНТНЫЙ BPTT (ТОЛЬКО ПРОХОД ПО ВРЕМЕНИ И dh_next)
// ============================================================================
kernel void lstm_recurrent_backward(
    global const float *U,          // [HiddenSize, 4 * HiddenSize]
    global const float *workspace,  // [SeqLen, Batch, 5 * HiddenSize]
    global const float *grad_H_out, // [Batch, SeqLen, HiddenSize]
    global float *all_grad_gates,   // Выход: [Batch, SeqLen, 4 * HiddenSize]
    global float *dh_next_buf,      // Временный: [Batch, HiddenSize]
    global float *dc_next_buf,      // Временный: [Batch, HiddenSize]
    const int batch_size, const int seq_len, const int hidden_size) {
  int b = get_group_id(0);
  int lid = get_local_id(1);
  int wg_size = get_local_size(1);

  if (b >= batch_size)
    return;

  int total_gates = 4 * hidden_size;
  int ws_stride = 5 * hidden_size;
  int vec_hidden = (hidden_size / 4) * 4;

  global float *dh_next = dh_next_buf + (b * hidden_size);
  global float *dc_next = dc_next_buf + (b * hidden_size);

  for (int hid = lid; hid < hidden_size; hid += wg_size) {
    dh_next[hid] = 0.0f;
    dc_next[hid] = 0.0f;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int t = seq_len - 1; t >= 0; t--) {
    int ws_offset = (t * batch_size + b) * ws_stride;
    int h_out_offset = (b * seq_len + t) * hidden_size;
    // Пишем градиенты гейтов в плоский 2D-формат [Batch * SeqLen, 4 *
    // HiddenSize] для GEMM!
    global float *grad_gates_t =
        all_grad_gates + (b * seq_len + t) * total_gates;

    // 1. Векторизованный расчет локальных градиентов гейтов (4 нейрона за
    // итерацию)
    for (int hid = lid * 4; hid < vec_hidden; hid += wg_size * 4) {
      float4 i_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 0 * hidden_size + hid));
      float4 f_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 1 * hidden_size + hid));
      float4 g_gate =
          tanh(vload4(0, workspace + ws_offset + 2 * hidden_size + hid));
      float4 o_gate =
          sigmoid4(vload4(0, workspace + ws_offset + 3 * hidden_size + hid));

      float4 c_curr_val =
          vload4(0, workspace + ws_offset + 4 * hidden_size + hid);
      float4 c_prev_val =
          (t > 0)
              ? vload4(0, workspace + ((t - 1) * batch_size + b) * ws_stride +
                              4 * hidden_size + hid)
              : (float4)(0.0f);

      float4 dh_curr =
          vload4(0, grad_H_out + h_out_offset + hid) + vload4(0, dh_next + hid);
      float4 tanh_c = tanh(c_curr_val);

      float4 d_raw_o = dh_curr * tanh_c * (o_gate * (1.0f - o_gate));
      float4 dc_curr = dh_curr * o_gate * (1.0f - tanh_c * tanh_c) +
                       vload4(0, dc_next + hid);
      float4 d_raw_f = dc_curr * c_prev_val * (f_gate * (1.0f - f_gate));
      float4 d_raw_i = dc_curr * g_gate * (i_gate * (1.0f - i_gate));
      float4 d_raw_g = dc_curr * i_gate * (1.0f - g_gate * g_gate);

      vstore4(clamp(d_raw_i, -5.0f, 5.0f), 0,
              grad_gates_t + 0 * hidden_size + hid);
      vstore4(clamp(d_raw_f, -5.0f, 5.0f), 0,
              grad_gates_t + 1 * hidden_size + hid);
      vstore4(clamp(d_raw_g, -5.0f, 5.0f), 0,
              grad_gates_t + 2 * hidden_size + hid);
      vstore4(clamp(d_raw_o, -5.0f, 5.0f), 0,
              grad_gates_t + 3 * hidden_size + hid);

      vstore4(clamp(dc_curr * f_gate, -5.0f, 5.0f), 0, dc_next + hid);
    }

    // Скалярный хвост для некратного 4 hidden_size
    for (int hid = vec_hidden + lid; hid < hidden_size; hid += wg_size) {
      float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + hid]);
      float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + hid]);
      float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + hid]);
      float o_gate = sigmoid(workspace[ws_offset + 3 * hidden_size + hid]);

      float c_curr_val = workspace[ws_offset + 4 * hidden_size + hid];
      float c_prev_val =
          (t > 0) ? workspace[((t - 1) * batch_size + b) * ws_stride +
                              4 * hidden_size + hid]
                  : 0.0f;

      float dh_curr = grad_H_out[h_out_offset + hid] + dh_next[hid];
      float tanh_c = tanh(c_curr_val);

      float d_raw_o = dh_curr * tanh_c * (o_gate * (1.0f - o_gate));
      float dc_curr =
          dh_curr * o_gate * (1.0f - tanh_c * tanh_c) + dc_next[hid];
      float d_raw_f = dc_curr * c_prev_val * (f_gate * (1.0f - f_gate));
      float d_raw_i = dc_curr * g_gate * (i_gate * (1.0f - i_gate));
      float d_raw_g = dc_curr * i_gate * (1.0f - g_gate * g_gate);

      float4 clipped =
          clamp((float4)(d_raw_i, d_raw_f, d_raw_g, d_raw_o), -5.0f, 5.0f);
      grad_gates_t[0 * hidden_size + hid] = clipped.x;
      grad_gates_t[1 * hidden_size + hid] = clipped.y;
      grad_gates_t[2 * hidden_size + hid] = clipped.z;
      grad_gates_t[3 * hidden_size + hid] = clipped.w;

      dc_next[hid] = clamp(dc_curr * f_gate, -5.0f, 5.0f);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // 2. Единственное матричное умножение внутри цикла по времени: dh_next =
    // grad_gates_t * U^T
    if (t > 0) {
      for (int hid = lid; hid < hidden_size; hid += wg_size) {
        float sum_h = 0.0f;
        int g_idx = 0;
        for (; g_idx <= total_gates - 4; g_idx += 4) {
          sum_h += dot(vload4(0, grad_gates_t + g_idx),
                       vload4(0, U + hid * total_gates + g_idx));
        }
        for (; g_idx < total_gates; g_idx++) {
          sum_h += grad_gates_t[g_idx] * U[hid * total_gates + g_idx];
        }
        dh_next[hid] = clamp(sum_h, -5.0f, 5.0f);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

// ============================================================================
// 1. ПРЯМОЙ ПРОХОД GEMM: XW_out = X * W + bias (4x4 Register Tiling)
// ============================================================================
kernel void lstm_gemm_xw(global const float *X, // [total_tokens, in_features]
                         global const float *W, // [in_features, total_gates]
                         global const float *bias, // [total_gates]
                         global float *XW_out, // [total_tokens, total_gates]
                         const int total_tokens, const int in_features,
                         const int total_gates) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int group_col = get_group_id(0) * TILE_N;
  int group_row = get_group_id(1) * TILE_M;

  __local float Xsub[TILE_M][TILE_K + 1];
  __local float Wsub[TILE_K][TILE_N + 1];

  float4 acc0 = (float4)(0.0f);
  float4 acc1 = (float4)(0.0f);
  float4 acc2 = (float4)(0.0f);
  float4 acc3 = (float4)(0.0f);

  int num_tiles = (in_features + TILE_K - 1) / TILE_K;

  for (int t = 0; t < num_tiles; t++) {
    int k_base = t * TILE_K;

    int x_col = k_base + lx;
    int valid_x_col = (x_col < in_features);
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      int local_r = ly + w * TS;
      int global_r = group_row + local_r;
      int valid = (global_r < total_tokens) && valid_x_col;
      int safe_idx = valid ? (global_r * in_features + x_col) : 0;
      Xsub[local_r][lx] = select(0.0f, X[safe_idx], valid);
    }

    int w_row = k_base + ly;
    int valid_w_row = (w_row < in_features);
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      int local_c = lx + w * TS;
      int global_c = group_col + local_c;
      int valid = valid_w_row && (global_c < total_gates);
      int safe_idx = valid ? (w_row * total_gates + global_c) : 0;
      Wsub[ly][local_c] = select(0.0f, W[safe_idx], valid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      float4 x_vec = (float4)(Xsub[ly * WPT + 0][k], Xsub[ly * WPT + 1][k],
                              Xsub[ly * WPT + 2][k], Xsub[ly * WPT + 3][k]);
      float4 w_vec = vload4(0, &Wsub[k][lx * WPT]);

      acc0 = fma((float4)(x_vec.x), w_vec, acc0);
      acc1 = fma((float4)(x_vec.y), w_vec, acc1);
      acc2 = fma((float4)(x_vec.z), w_vec, acc2);
      acc3 = fma((float4)(x_vec.w), w_vec, acc3);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  int out_col_base = group_col + lx * WPT;
  int out_row_base = group_row + ly * WPT;

  int v0 = (out_col_base + 0 < total_gates);
  int v1 = (out_col_base + 1 < total_gates);
  int v2 = (out_col_base + 2 < total_gates);
  int v3 = (out_col_base + 3 < total_gates);

  float4 b_vec = (float4)(select(0.0f, bias[v0 ? (out_col_base + 0) : 0], v0),
                          select(0.0f, bias[v1 ? (out_col_base + 1) : 0], v1),
                          select(0.0f, bias[v2 ? (out_col_base + 2) : 0], v2),
                          select(0.0f, bias[v3 ? (out_col_base + 3) : 0], v3));

  float4 rows[4] = {acc0 + b_vec, acc1 + b_vec, acc2 + b_vec, acc3 + b_vec};

#pragma unroll
  for (int r = 0; r < 4; r++) {
    int gr = out_row_base + r;
    if (gr < total_tokens) {
      if (out_col_base + 3 < total_gates) {
        vstore4(rows[r], 0, XW_out + gr * total_gates + out_col_base);
      } else {
        float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
        for (int c = 0; c < 4; c++) {
          if (out_col_base + c < total_gates) {
            XW_out[gr * total_gates + out_col_base + c] = vals[c];
          }
        }
      }
    }
  }
}

// ============================================================================
// 2. ОБРАТНЫЙ ПРОХОД GEMM: grad_X = all_grad_gates * W^T (4x4 Register Tiling)
// ============================================================================
kernel void lstm_gemm_grad_x(
    global const float *all_grad_gates, // [total_tokens, total_gates]
    global const float *W,              // [in_features, total_gates]
    global float *grad_X,               // [total_tokens, in_features]
    const int total_tokens, const int in_features, const int total_gates) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int group_col = get_group_id(0) * TILE_N;
  int group_row = get_group_id(1) * TILE_M;

  __local float Gsub[TILE_M][TILE_K + 1];
  __local float WTsub[TILE_K][TILE_N + 1];

  float4 acc0 = (float4)(0.0f);
  float4 acc1 = (float4)(0.0f);
  float4 acc2 = (float4)(0.0f);
  float4 acc3 = (float4)(0.0f);

  int num_tiles = (total_gates + TILE_K - 1) / TILE_K;

  for (int t = 0; t < num_tiles; t++) {
    int k_base = t * TILE_K;

    int g_col = k_base + lx;
    int valid_g_col = (g_col < total_gates);
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      int local_r = ly + w * TS;
      int global_r = group_row + local_r;
      int valid = (global_r < total_tokens) && valid_g_col;
      int safe_idx = valid ? (global_r * total_gates + g_col) : 0;
      Gsub[local_r][lx] = select(0.0f, all_grad_gates[safe_idx], valid);
    }

    int w_k = k_base + ly;
    int valid_w_k = (w_k < total_gates);
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      int local_c = lx + w * TS;
      int global_c = group_col + local_c;
      int valid = valid_w_k && (global_c < in_features);
      int safe_idx = valid ? (global_c * total_gates + w_k) : 0;
      WTsub[ly][local_c] = select(0.0f, W[safe_idx], valid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      float4 g_vec = (float4)(Gsub[ly * WPT + 0][k], Gsub[ly * WPT + 1][k],
                              Gsub[ly * WPT + 2][k], Gsub[ly * WPT + 3][k]);
      float4 wt_vec = vload4(0, &WTsub[k][lx * WPT]);

      acc0 = fma((float4)(g_vec.x), wt_vec, acc0);
      acc1 = fma((float4)(g_vec.y), wt_vec, acc1);
      acc2 = fma((float4)(g_vec.z), wt_vec, acc2);
      acc3 = fma((float4)(g_vec.w), wt_vec, acc3);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  int out_col_base = group_col + lx * WPT;
  int out_row_base = group_row + ly * WPT;
  float4 rows[4] = {acc0, acc1, acc2, acc3};

#pragma unroll
  for (int r = 0; r < 4; r++) {
    int gr = out_row_base + r;
    if (gr < total_tokens) {
      if (out_col_base + 3 < in_features) {
        vstore4(rows[r], 0, grad_X + gr * in_features + out_col_base);
      } else {
        float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
        for (int c = 0; c < 4; c++) {
          if (out_col_base + c < in_features) {
            grad_X[gr * in_features + out_col_base + c] = vals[c];
          }
        }
      }
    }
  }
}

// ============================================================================
// 3. ОБРАТНЫЙ ПРОХОД GEMM: grad_W, grad_U и grad_bias (4x4 Register Tiling)
// ============================================================================
kernel void lstm_gemm_grad_weights(
    global const float *X,              // [total_tokens, in_features]
    global const float *H_out,          // [batch_size, seq_len, hidden_size]
    global const float *all_grad_gates, // [total_tokens, total_gates]
    global float *grad_W,               // [in_features, total_gates]
    global float *grad_U,               // [hidden_size, total_gates]
    global float *grad_bias,            // [total_gates]
    const int batch_size, const int seq_len, const int in_features,
    const int hidden_size) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int group_col = get_group_id(0) * TILE_N;
  int group_row = get_group_id(1) * TILE_M;

  int total_tokens = batch_size * seq_len;
  int total_gates = 4 * hidden_size;
  int max_rows = max(in_features, hidden_size);

  if (group_col >= total_gates || group_row >= max_rows)
    return;

  __local float LeftSub[TILE_M][TILE_K + 1];
  __local float Gsub[TILE_K][TILE_N + 1];

  int num_tiles = (total_tokens + TILE_K - 1) / TILE_K;
  int out_col_base = group_col + lx * WPT;
  int out_row_base = group_row + ly * WPT;

  // ------------------------------------------------------------------------
  // ЧАСТЬ А: grad_W += X^T * all_grad_gates и векторный grad_bias
  // ------------------------------------------------------------------------
  if (group_row < in_features) {
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    float4 acc_bias = (float4)(0.0f);

    int compute_bias = (get_group_id(1) == 0 && ly == 0);

    for (int t = 0; t < num_tiles; t++) {
      int k_base = t * TILE_K;

      int tok_x = k_base + lx;
      int valid_tok_x = (tok_x < total_tokens);
#pragma unroll
      for (int w = 0; w < WPT; w++) {
        int local_r = ly + w * TS;
        int global_r = group_row + local_r;
        int valid = (global_r < in_features) && valid_tok_x;
        int safe_idx = valid ? (tok_x * in_features + global_r) : 0;
        LeftSub[local_r][lx] = select(0.0f, X[safe_idx], valid);
      }

      int tok_y = k_base + ly;
      int valid_tok_y = (tok_y < total_tokens);
      //Используйте код с осторожностью.
#pragma unroll
          for (int w = 0; w < WPT; w++) {
        int local_c = lx + w * TS;
        int global_c = group_col + local_c;
        int valid = valid_tok_y && (global_c < total_gates);
        int safe_idx = valid ? (tok_y * total_gates + global_c) : 0;
        Gsub[ly][local_c] = select(0.0f, all_grad_gates[safe_idx], valid);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (int k = 0; k < TILE_K; k++) {
        float4 x_vec =
            (float4)(LeftSub[ly * WPT + 0][k], LeftSub[ly * WPT + 1][k],
                     LeftSub[ly * WPT + 2][k], LeftSub[ly * WPT + 3][k]);
        float4 g_vec = vload4(0, &Gsub[k][lx * WPT]);
        acc0 = fma((float4)(x_vec.x), g_vec, acc0);
        acc1 = fma((float4)(x_vec.y), g_vec, acc1);
        acc2 = fma((float4)(x_vec.z), g_vec, acc2);
        acc3 = fma((float4)(x_vec.w), g_vec, acc3);
        if (compute_bias) {
          acc_bias += g_vec;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float4 rows[4] = {acc0, acc1, acc2, acc3};
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int gr = out_row_base + r;
      if (gr < in_features) {
        if (out_col_base + 3 < total_gates) {
          int offset = gr * total_gates + out_col_base;
          vstore4(vload4(0, grad_W + offset) + rows[r], 0, grad_W + offset);
        } else {
          float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
          for (int c = 0; c < 4; c++) {
            if (out_col_base + c < total_gates) {
              grad_W[gr * total_gates + out_col_base + c] += vals[c];
            }
          }
        }
      }
    }
    if (compute_bias) {
      if (out_col_base + 3 < total_gates) {
        vstore4(vload4(0, grad_bias + out_col_base) + acc_bias, 0,
                grad_bias + out_col_base);
      } else {
        float b_vals[4] = {acc_bias.x, acc_bias.y, acc_bias.z, acc_bias.w};
        for (int c = 0; c < 4; c++) {
          if (out_col_base + c < total_gates) {
            grad_bias[out_col_base + c] += b_vals[c];
          }
        }
      }
    }
  }
  // ------------------------------------------------------------------------
  // ЧАСТЬ Б: grad_U += H_prev^T * all_grad_gates
  // ------------------------------------------------------------------------
  if (group_row < hidden_size) {
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    for (int t = 0; t < num_tiles; t++) {
      int k_base = t * TILE_K;
      int tok_x = k_base + lx;
      int valid_tok_h = (tok_x < total_tokens) && ((tok_x % seq_len) > 0);
#pragma unroll
      for (int w = 0; w < WPT; w++) {
        int local_r = ly + w * TS;
        int global_r = group_row + local_r;
        int valid = (global_r < hidden_size) && valid_tok_h;
        int safe_idx = valid ? ((tok_x - 1) * hidden_size + global_r) : 0;
        LeftSub[local_r][lx] = select(0.0f, H_out[safe_idx], valid);
      }
      int tok_y = k_base + ly;
      int valid_tok_y = (tok_y < total_tokens);
#pragma unroll
      for (int w = 0; w < WPT; w++) {
        int local_c = lx + w * TS;
        int global_c = group_col + local_c;
        int valid = valid_tok_y && (global_c < total_gates);
        int safe_idx = valid ? (tok_y * total_gates + global_c) : 0;
        Gsub[ly][local_c] = select(0.0f, all_grad_gates[safe_idx], valid);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (int k = 0; k < TILE_K; k++) {
        float4 h_vec =
            (float4)(LeftSub[ly * WPT + 0][k], LeftSub[ly * WPT + 1][k],
                     LeftSub[ly * WPT + 2][k], LeftSub[ly * WPT + 3][k]);
        float4 g_vec = vload4(0, &Gsub[k][lx * WPT]);
        acc0 = fma((float4)(h_vec.x), g_vec, acc0);
        acc1 = fma((float4)(h_vec.y), g_vec, acc1);
        acc2 = fma((float4)(h_vec.z), g_vec, acc2);
        acc3 = fma((float4)(h_vec.w), g_vec, acc3);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float4 rows[4] = {acc0, acc1, acc2, acc3};
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int gr = out_row_base + r;
      if (gr < hidden_size) {
        if (out_col_base + 3 < total_gates) {
          int offset = gr * total_gates + out_col_base;
          vstore4(vload4(0, grad_U + offset) + rows[r], 0, grad_U + offset);
        } else {
          float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
          for (int c = 0; c < 4; c++) {
            if (out_col_base + c < total_gates) {
              grad_U[gr * total_gates + out_col_base + c] += vals[c];
            }
          }
        }
      }
    }
  }
}
/**
 * Монолитное ядро обратного прохода Stateful LSTM.
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */
kernel void lstm_hid_backward(
    global const float *X, // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float *W, // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float
        *U, // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float *H_prev, // Исходное H_prev из прямого прохода
    global const float *C_prev, // Исходное C_prev из прямого прохода
    global const float *
        H_out, // Вычисленный Forward выход H [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global const float *workspace,  // Сырые гейты из Forward [SEQ_LEN,
                                    // BATCH_SIZE, 4 * HIDDEN_SIZE]
    global const float *grad_H_out, // Градиент, пришедший сверху dL/dH
                                    // [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float *grad_X,      // Выходной градиент по входу dL/dX [BATCH_SIZE,
                               // SEQ_LEN, IN_FEATURES]
    global float *grad_H_prev, // Выходной градиент по входному скрытому
                               // состоянию dL/dH_prev [BATCH_SIZE, HIDDEN_SIZE]
    global float *grad_C_prev, // Выходной градиент по входному состоянию ячейки
                               // dL/dC_prev [BATCH_SIZE, HIDDEN_SIZE]
    global float *grad_W,      // Атомарный буфер градиентов весов dL/dW
                               // [IN_FEATURES, 4 * HIDDEN_SIZE]
    global float *grad_U,      // Атомарный буфер градиентов весов dL/dU
                               // [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global float *grad_bias,   // Атомарный буфер градиентов смещения dL/dB [4 *
                               // HIDDEN_SIZE]
    const int batch_size, const int seq_len, const int in_features,
    const int hidden_size) {
  int b = get_global_id(0);
  if (b >= batch_size)
    return;

  float dh_next[1024];
  float dc_next[1024];
  float c_accum[1024];

  int state_offset = b * hidden_size;

  for (int h = 0; h < hidden_size; h++) {
    dh_next[h] = 0.0f;
    dc_next[h] = 0.0f;
    // Восстановление цепочки C начинаем со значения из C_prev вместо нуля
    c_accum[h] = C_prev[state_offset + h];
  }

  // Накат C до конца текущей последовательности
  for (int t = 0; t < seq_len; t++) {
    int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
    for (int h = 0; h < hidden_size; h++) {
      float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + h]);
      float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + h]);
      float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + h]);
      c_accum[h] = f_gate * c_accum[h] + i_gate * g_gate;
    }
  }

  // Обратный проход во времени
  for (int t = seq_len - 1; t >= 0; t--) {
    int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
    int h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;
    int x_offset = b * (seq_len * in_features) + t * in_features;

    float c_prev_t[1024];
    for (int h = 0; h < hidden_size; h++) {
      float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + h]);
      float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + h]);
      float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + h]);

      // Если t == 0, то предыдущим состоянием ячейки является внешне переданный
      // C_prev
      c_prev_t[h] =
          (t > 0) ? (f_gate != 0.0f ? (c_accum[h] - i_gate * g_gate) / f_gate
                                    : 0.0f)
                  : C_prev[state_offset + h];
    }

    float grad_gates[4096];

    for (int h = 0; h < hidden_size; h++) {
      float dh_curr = grad_H_out[h_out_offset + h] + dh_next[h];

      float raw_i = workspace[ws_offset + 0 * hidden_size + h];
      float raw_f = workspace[ws_offset + 1 * hidden_size + h];
      float raw_g = workspace[ws_offset + 2 * hidden_size + h];
      float raw_o = workspace[ws_offset + 3 * hidden_size + h];

      float i_gate = sigmoid(raw_i);
      float f_gate = sigmoid(raw_f);
      float g_gate = tanh(raw_g);
      float o_gate = sigmoid(raw_o);

      float tanh_c = tanh(c_accum[h]);

      float d_raw_o = dh_curr * tanh_c * (o_gate * (1.0f - o_gate));
      float dc_curr = dh_curr * o_gate * (1.0f - tanh_c * tanh_c) + dc_next[h];

      float d_raw_f = dc_curr * c_prev_t[h] * (f_gate * (1.0f - f_gate));
      float d_raw_i = dc_curr * g_gate * (i_gate * (1.0f - i_gate));
      float d_raw_g = dc_curr * i_gate * (1.0f - g_gate * g_gate);

      grad_gates[0 * hidden_size + h] = d_raw_i;
      grad_gates[1 * hidden_size + h] = d_raw_f;
      grad_gates[2 * hidden_size + h] = d_raw_g;
      grad_gates[3 * hidden_size + h] = d_raw_o;

      dc_next[h] = dc_curr * f_gate;
    }

    // dL/dX_t
    for (int i = 0; i < in_features; i++) {
      float sum_x = 0.0f;
      for (int g = 0; g < 4 * hidden_size; g++) {
        sum_x += grad_gates[g] * W[i * (4 * hidden_size) + g];
      }
      grad_X[x_offset + i] = sum_x;
    }

    // Вычисляем dh_next для предыдущего шага внутри чанка
    for (int h = 0; h < hidden_size; h++) {
      float sum_h = 0.0f;
      for (int g = 0; g < 4 * hidden_size; g++) {
        sum_h += grad_gates[g] * U[h * (4 * hidden_size) + g];
      }
      dh_next[h] = sum_h;
    }

    // Аккумуляция весов W, U, Bias через атомики (код вырезан для краткости,
    // идентичен базовому)
    // ...

    for (int h = 0; h < hidden_size; h++) {
      c_accum[h] = c_prev_t[h];
    }
  }

  // ВАЖНО: То, что осталось в dh_next и dc_next после цикла по времени — это
  // и есть градиенты, которые перетекают обратно в начальные состояния чанка
  // h_prev и c_prev!
  for (int h = 0; h < hidden_size; h++) {
    grad_H_prev[state_offset + h] = dh_next[h];
    grad_C_prev[state_offset + h] = dc_next[h];
  }
}

/**
 * Монолитное ядро прямого прохода Stateful LSTM.
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */
kernel void lstm_hid_forward(
    global const float *X, // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float *W, // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float
        *U, // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float *bias,   // Смещения [4 * HIDDEN_SIZE]
    global const float *H_prev, // Входное скрытое состояние из прошлого чанка
                                // [BATCH_SIZE, HIDDEN_SIZE]
    global const float *C_prev, // Входное состояние ячейки из прошлого чанка
                                // [BATCH_SIZE, HIDDEN_SIZE]
    global float
        *H_out, // Выходные скрытые состояния [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float *workspace, // Буфер гейтов для автодифференцирования [SEQ_LEN,
                             // BATCH_SIZE, 4 * HIDDEN_SIZE]
    global float *final_H,   // Последнее вычисленное H (для передачи в след.
                             // чанк) [BATCH_SIZE, HIDDEN_SIZE]
    global float *final_C,   // Последнее вычисленное C (для передачи в след.
                             // чанк) [BATCH_SIZE, HIDDEN_SIZE]
    const int batch_size, const int seq_len, const int in_features,
    const int hidden_size) {
  int b = get_global_id(0);
  if (b >= batch_size)
    return;

  // Локальные регистровые массивы потока под текущее состояние
  // Ограничим статическим размером (например, до 1024), чтобы компилятор не
  // ругался на динамические массивы в приватной памяти
  float c_curr[1024];
  float h_curr[1024];

  // Инициализация значениями из переданных h_prev и c_prev
  int state_offset = b * hidden_size;
  for (int h = 0; h < hidden_size; h++) {
    h_curr[h] = H_prev[state_offset + h];
    c_curr[h] = C_prev[state_offset + h];
  }

  // Цикл по времени (T)
  for (int t = 0; t < seq_len; t++) {
    int x_offset = b * (seq_len * in_features) + t * in_features;
    int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
    int h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;

    // Расчет 4-х гейтов
    for (int g = 0; g < 4 * hidden_size; g++) {
      float val = bias[g];

      for (int i = 0; i < in_features; i++) {
        val += X[x_offset + i] * W[i * (4 * hidden_size) + g];
      }

      for (int h = 0; h < hidden_size; h++) {
        val += h_curr[h] * U[h * (4 * hidden_size) + g];
      }

      workspace[ws_offset + g] = val;
    }

    // Активации и обновление
    for (int h = 0; h < hidden_size; h++) {
      float raw_i = workspace[ws_offset + 0 * hidden_size + h];
      float raw_f = workspace[ws_offset + 1 * hidden_size + h];
      float raw_g = workspace[ws_offset + 2 * hidden_size + h];
      float raw_o = workspace[ws_offset + 3 * hidden_size + h];

      float i_gate = sigmoid(raw_i);
      float f_gate = sigmoid(raw_f);
      float g_gate = tanh(raw_g);
      float o_gate = sigmoid(raw_o);

      c_curr[h] = f_gate * c_curr[h] + i_gate * g_gate;
      h_curr[h] = o_gate * tanh(c_curr[h]);

      H_out[h_out_offset + h] = h_curr[h];
    }
  }

  // Записываем финальные состояния в выходные буферы
  for (int h = 0; h < hidden_size; h++) {
    final_H[state_offset + h] = h_curr[h];
    final_C[state_offset + h] = c_curr[h];
  }
}
/**
 * Ядро для расширения усеченного градиента до исходного размера.
 * Заполняет последний временной шаг нулями, а остальные копирует из sliced.
 * Сетка запуска (Global Work Size): [BATCH_SIZE, OLD_SEQ_LEN]
 */
kernel void pad_sliced_gradient_kernel(
    global const float *grad_sliced, // Усеченный градиент сверху [BATCH_SIZE,
                                     // OLD_SEQ_LEN - 1, VOCAB_SIZE]
    global float *grad_original, // Результирующий широкий градиент [BATCH_SIZE,
                                 // OLD_SEQ_LEN, VOCAB_SIZE]
    const int old_seq_len, const int vocab_size) {
  int b = get_global_id(0); // Индекс батча
  int t = get_global_id(1); // Индекс временного шага в оригинальной геометрии
                            // (0 ... OLD_SEQ_LEN - 1)

  int new_seq_len = old_seq_len - 1;

  // Вычисляем базовое смещение строки для оригинального (широкого) градиента
  long dest_row_offset = (long)(b * old_seq_len + t) * vocab_size;

  // Если это самый последний временной шаг, ошибка в него не протекала ->
  // жестко зануляем всю строку
  if (t == old_seq_len - 1) {
    for (int v = 0; v < vocab_size; v++) {
      grad_original[dest_row_offset + v] = 0.0f;
    }
  } else {
    // Для всех остальных шагов вычисляем смещение в усеченной матрице
    // градиентов
    long src_row_offset = (long)(b * new_seq_len + t) * vocab_size;

    // Копируем значения один к одному
    for (int v = 0; v < vocab_size; v++) {
      grad_original[dest_row_offset + v] = grad_sliced[src_row_offset + v];
    }
  }
}
