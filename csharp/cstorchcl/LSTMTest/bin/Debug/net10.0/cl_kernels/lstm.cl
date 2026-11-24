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

#define MAX_LOCAL_HIDDEN 2048
// Межблочный барьер для K рабочих групп одного батча b
inline void global_batch_sync(
    global volatile int
        *batch_ctrl,        // Указатель на controller[b] (изначально 0)
    const int num_groups_x, // K: количество рабочих групп на 1 батч
    const int lid,          // Локальный ID потока внутри группы
    const int step          // Монотонный номер шага синхронизации (1, 2, 3...)
) {
  // 1. Все потоки текущей группы завершают запись в глобальную память и
  // сбрасывают L1-кэш
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // 2. Только лидер группы (lid == 0) отмечается в глобальном контроллере
  if (lid == 0 && num_groups_x > 1) {
    int target = step * num_groups_x;
    atomic_inc(batch_ctrl);

    // Ждем, пока все K групп дойдут до этого шага (чтение мимо L1-кэша!)
    while (atomic_add(batch_ctrl, 0) < target) {
      // Активное ожидание
    }
  }

  // 3. Остальные потоки группы ждут лидера и обновляют видимость глобальной
  // памяти
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void lstm_recurrent_forward(
    global const float *XW_in,       // [Batch, SeqLen, 4 * HiddenSize]
    global const float *U,           // [HiddenSize, 4 * HiddenSize]
    global float *H_out,             // [Batch, SeqLen, HiddenSize]
    global float *workspace,         // [SeqLen, Batch, 5 * HiddenSize]
    global float *c_curr_buf,        // [Batch, HiddenSize]
    global float *h_curr_buf,        // [Batch, HiddenSize]
    global volatile int *controller, // [Batch] — обнулен перед запуском!
    const int batch_size, const int seq_len, const int hidden_size) {
  int gid_x = get_global_id(0);
  int total_x = get_global_size(0); // K * wg_size
  int lid = get_local_id(0);
  int num_groups_x = get_num_groups(0); // K блоков на батч
  int b = get_group_id(2);              // Ось Z = индекс батча

  if (b >= batch_size)
    return;

  global float *c_curr_ptr = c_curr_buf + (b * hidden_size);
  global float *h_curr_ptr = h_curr_buf + (b * hidden_size);
  global volatile int *batch_ctrl = controller + b;

  int sync_step = 0;

  // Инициализация состояний всеми K группами батча
  for (int hid = gid_x; hid < hidden_size; hid += total_x) {
    c_curr_ptr[hid] = 0.0f;
    h_curr_ptr[hid] = 0.0f;
  }
  global_batch_sync(batch_ctrl, num_groups_x, lid, ++sync_step);

  int total_gates = 4 * hidden_size;
  int ws_stride = 5 * hidden_size;

  for (int t = 0; t < seq_len; t++) {
    int xw_offset = (b * seq_len + t) * total_gates;
    int ws_offset = (t * batch_size + b) * ws_stride;
    int h_out_offset = (b * seq_len + t) * hidden_size;

    // K * 256 потоков делят между собой нейроны hid
    for (int hid = gid_x; hid < hidden_size; hid += total_x) {
      int idx_i = 0 * hidden_size + hid;
      int idx_f = 1 * hidden_size + hid;
      int idx_g = 2 * hidden_size + hid;
      int idx_o = 3 * hidden_size + hid;

      float4 acc = (float4)(XW_in[xw_offset + idx_i], XW_in[xw_offset + idx_f],
                            XW_in[xw_offset + idx_g], XW_in[xw_offset + idx_o]);

      int h = 0;
      for (; h <= hidden_size - 4; h += 4) {
        float4 h_vec = vload4(0, h_curr_ptr + h);
        int r0 = (h + 0) * total_gates;
        int r1 = (h + 1) * total_gates;
        int r2 = (h + 2) * total_gates;
        int r3 = (h + 3) * total_gates;

        acc = fma((float4)(h_vec.x),
                  (float4)(U[r0 + idx_i], U[r0 + idx_f], U[r0 + idx_g],
                           U[r0 + idx_o]),
                  acc);
        acc = fma((float4)(h_vec.y),
                  (float4)(U[r1 + idx_i], U[r1 + idx_f], U[r1 + idx_g],
                           U[r1 + idx_o]),
                  acc);
        acc = fma((float4)(h_vec.z),
                  (float4)(U[r2 + idx_i], U[r2 + idx_f], U[r2 + idx_g],
                           U[r2 + idx_o]),
                  acc);
        acc = fma((float4)(h_vec.w),
                  (float4)(U[r3 + idx_i], U[r3 + idx_f], U[r3 + idx_g],
                           U[r3 + idx_o]),
                  acc);
      }
      for (; h < hidden_size; h++) {
        int u_row = h * total_gates;
        acc = fma((float4)(h_curr_ptr[h]),
                  (float4)(U[u_row + idx_i], U[u_row + idx_f], U[u_row + idx_g],
                           U[u_row + idx_o]),
                  acc);
      }

      workspace[ws_offset + idx_i] = acc.x;
      workspace[ws_offset + idx_f] = acc.y;
      workspace[ws_offset + idx_g] = acc.z;
      workspace[ws_offset + idx_o] = acc.w;

      float i_gate = sigmoid(acc.x);
      float f_gate = sigmoid(acc.y);
      float g_gate = tanh(acc.z);
      float o_gate = sigmoid(acc.w);

      float c_val = f_gate * c_curr_ptr[hid] + i_gate * g_gate;
      c_curr_ptr[hid] = c_val;
      workspace[ws_offset + 4 * hidden_size + hid] = c_val;

      float h_val = o_gate * tanh(c_val);
      h_curr_ptr[hid] = h_val;
      H_out[h_out_offset + hid] = h_val;
    }

    // Межблочная синхронизация всех K групп перед переходом к t + 1
    global_batch_sync(batch_ctrl, num_groups_x, lid, ++sync_step);
  }
}

// ============================================================================
// ОБРАТНЫЙ ПРОХОД (BACKWARD) БЕЗ АТОМИКОВ ВНУТРИ ЦИКЛА ПО ВРЕМЕНИ
// ============================================================================
// ============================================================================
// ЯДРО 1: ЧИСТЫЙ РЕКУРРЕНТНЫЙ BPTT (ТОЛЬКО ПРОХОД ПО ВРЕМЕНИ И dh_next)
// ============================================================================
kernel void lstm_recurrent_backward(
    global const float *U,           // [HiddenSize, 4 * HiddenSize]
    global const float *workspace,   // [SeqLen, Batch, 5 * HiddenSize]
    global const float *grad_H_out,  // [Batch, SeqLen, HiddenSize]
    global float *all_grad_gates,    // [Batch, SeqLen, 4 * HiddenSize]
    global float *dh_next_buf,       // [Batch, HiddenSize]
    global float *dc_next_buf,       // [Batch, HiddenSize]
    global volatile int *controller, // [Batch] — обнулен перед запуском!
    const int batch_size, const int seq_len, const int hidden_size) {
  int gid_x = get_global_id(0);
  int total_x = get_global_size(0);
  int lid = get_local_id(0);
  int num_groups_x = get_num_groups(0);
  int b = get_group_id(2);

  if (b >= batch_size)
    return;

  int total_gates = 4 * hidden_size;
  int ws_stride = 5 * hidden_size;

  global float *dh_next = dh_next_buf + (b * hidden_size);
  global float *dc_next = dc_next_buf + (b * hidden_size);
  global volatile int *batch_ctrl = controller + b;

  int sync_step = 0;

  for (int hid = gid_x; hid < hidden_size; hid += total_x) {
    dh_next[hid] = 0.0f;
    dc_next[hid] = 0.0f;
  }
  global_batch_sync(batch_ctrl, num_groups_x, lid, ++sync_step);

  for (int t = seq_len - 1; t >= 0; t--) {
    int ws_offset = (t * batch_size + b) * ws_stride;
    int h_out_offset = (b * seq_len + t) * hidden_size;
    global float *grad_gates_t =
        all_grad_gates + (b * seq_len + t) * total_gates;

    // 1. Все K групп считают локальные градиенты гейтов своего диапазона hid
    for (int hid = gid_x; hid < hidden_size; hid += total_x) {
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

    // Ждем, пока все K групп запишут свои куски grad_gates_t
    global_batch_sync(batch_ctrl, num_groups_x, lid, ++sync_step);

    // 2. Все K групп параллельно считают dh_next = grad_gates_t * U^T
    if (t > 0) {
      for (int hid = gid_x; hid < hidden_size; hid += total_x) {
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

      // Ждем, пока все K групп обновят dh_next перед шагом t - 1
      global_batch_sync(batch_ctrl, num_groups_x, lid, ++sync_step);
    }
  }
}

// ============================================================================
// 1. ПРЯМОЙ ПРОХОД GEMM: XW_out = X * W + bias (4x4 Register Tiling)
// ============================================================================



kernel void lstm_gemm_grad_x(
    global const float *all_grad_gates, // [total_tokens, total_gates]
    global const float *W,              // [in_features, total_gates]
    global float *grad_X,               // [total_tokens, in_features]
    const int total_tokens, const int in_features, const int total_gates) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int group_col = get_group_id(0) * TILE_N; // по оси in_features
  int group_row = get_group_id(1) * TILE_M; // по оси total_tokens

  __local float Gsub[TILE_M][TILE_K + 1];
  __local float WTsub[TILE_K][TILE_N + 1];

  float4 acc0 = (float4)(0.0f);
  float4 acc1 = (float4)(0.0f);
  float4 acc2 = (float4)(0.0f);
  float4 acc3 = (float4)(0.0f);

  int num_tiles = (total_gates + TILE_K - 1) / TILE_K;

  for (int t = 0; t < num_tiles; t++) {
    int k_base = t * TILE_K;

    // Предвычисляем инвариантные по w смещения и флаги границ
    int g_col = k_base + lx;
    int valid_g_col = (g_col < total_gates);

    // Для коалесцированного чтения W: lx идет вдоль младшей оси (total_gates)!
    int w_gate_col = k_base + lx;
    int valid_w_gate = (w_gate_col < total_gates);

// Объединенный цикл загрузки Gsub [64x16] и транспонированного WTsub [16x64]
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      // 1. Загрузка Gsub[local_r][lx] (Coalesced чтение по lx)
      int local_r = ly + w * TS;
      int global_r = group_row + local_r;
      int valid_g = (global_r < total_tokens) && valid_g_col;
      int safe_g = valid_g ? (global_r * total_gates + g_col) : 0;
      Gsub[local_r][lx] = select(0.0f, all_grad_gates[safe_g], valid_g);

      // 2. Коалесцированная загрузка W и транспонирование на лету в
      // WTsub[lx][local_c]
      int local_c = ly + w * TS;
      int global_c = group_col + local_c;
      int valid_w = (global_c < in_features) && valid_w_gate;
      int safe_w = valid_w ? (global_c * total_gates + w_gate_col) : 0;
      WTsub[lx][local_c] = select(0.0f, W[safe_w], valid_w);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

// Векторное 4x4 умножение в регистрах
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

  int full_vec_valid = (out_col_base + 3 < in_features);

#pragma unroll
  for (int r = 0; r < 4; r++) {
    int gr = out_row_base + r;
    if (gr < total_tokens) {
      int row_offset = gr * in_features + out_col_base;
      if (full_vec_valid) {
        vstore4(rows[r], 0, grad_X + row_offset);
      } else {
        float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
        for (int c = 0; c < 4; c++) {
          if (out_col_base + c < in_features) {
            grad_X[row_offset + c] = vals[c];
          }
        }
      }
    }
  }
}

// ============================================================================
// 1. ПРЯМОЙ ПРОХОД GEMM: XW_out = X * W + bias (Объединенный цикл загрузки)
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

    int w_row = k_base + ly;
    int valid_w_row = (w_row < in_features);
    int w_row_offset = w_row * total_gates;
  
// Объединенный цикл загрузки Xsub [64x16] и Wsub [16x64]
#pragma unroll
    for (int w = 0; w < WPT; w++) {
      int stride = w * TS;

      // 1. Загрузка Xsub[local_r][lx] (Coalesced чтение по lx)
      int local_r = ly + stride;
      int global_r = group_row + local_r;
      int valid_x = (global_r < total_tokens) && valid_x_col;
      int safe_x = valid_x ? (global_r * in_features + x_col) : 0;
      Xsub[local_r][lx] = select(0.0f, X[safe_x], valid_x);

      // 2. Загрузка Wsub[ly][local_c] (Coalesced чтение по lx)
      int local_c = lx + stride;
      int global_c = group_col + local_c;
      int valid_w = valid_w_row && (global_c < total_gates);
      int safe_w = valid_w ? (w_row_offset + global_c) : 0;
      Wsub[ly][local_c] = select(0.0f, W[safe_w], valid_w);
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

    float4 rows[4] = { acc0 + b_vec, acc1 + b_vec, acc2 + b_vec, acc3 + b_vec };
    int full_vec_valid = (out_col_base + 3 < total_gates);

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int gr = out_row_base + r;
        if (gr < total_tokens) {
            int row_offset = gr * total_gates + out_col_base;
            if (full_vec_valid) {
                vstore4(rows[r], 0, XW_out + row_offset);
            } else {
                float vals[4] = { rows[r].x, rows[r].y, rows[r].z, rows[r].w };
                for (int c = 0; c < 4; c++) {
                    if (out_col_base + c < total_gates) {
                        XW_out[row_offset + c] = vals[c];
                    }
                }
            }
        }
    }

}

// ============================================================================
// 2. ОБРАТНЫЙ ПРОХОД GEMM: grad_W, grad_U и grad_bias (Coalesced X^T и H^T)
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
  int full_vec_valid = (out_col_base + 3 < total_gates);

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

      // Для X^T: поток ly выбирает токен, а lx читает подряд идущие признаки
      // in_features (Coalesced!)
      int tok_x = k_base + ly;
      int valid_tok_x = (tok_x < total_tokens);
      int x_row_offset = tok_x * in_features;

      int tok_y = k_base + ly;
      int valid_tok_y = (tok_y < total_tokens);
      int g_row_offset = tok_y * total_gates;

#pragma unroll
      for (int w = 0; w < WPT; w++) {
        int stride = w * TS;

        // 1. Коалесцированная загрузка X и транспонирование на лету в
        // LeftSub[local_r][ly]
        int local_r = lx + stride;
        int global_r = group_row + local_r;
        int valid_x = valid_tok_x && (global_r < in_features);
        int safe_x = valid_x ? (x_row_offset + global_r) : 0;
        LeftSub[local_r][ly] = select(0.0f, X[safe_x], valid_x);

        // 2. Коалесцированная загрузка Gsub[ly][local_c]
        int local_c = lx + stride;
        int global_c = group_col + local_c;
        int valid_g = valid_tok_y && (global_c < total_gates);
        int safe_g = valid_g ? (g_row_offset + global_c) : 0;
        Gsub[ly][local_c] = select(0.0f, all_grad_gates[safe_g], valid_g);
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
        int offset = gr * total_gates + out_col_base;
        if (full_vec_valid) {
          vstore4(vload4(0, grad_W + offset) + rows[r], 0, grad_W + offset);
        } else {
          float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
          for (int c = 0; c < 4; c++) {
            if (out_col_base + c < total_gates) {
              grad_W[offset + c] += vals[c];
            }
          }
        }
      }
    }

    if (compute_bias) {
      if (full_vec_valid) {
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
  //Используйте код с осторожностью.
      // ------------------------------------------------------------------------
      if (group_row < hidden_size) {
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    for (int t = 0; t < num_tiles; t++) {
      int k_base = t * TILE_K;
      // Для H_prev^T: поток ly выбирает токен, а lx читает подряд идущие
      // нейроны hidden_size (Coalesced!)
      int tok_h = k_base + ly;
      int valid_tok_h = (tok_h < total_tokens) && ((tok_h % seq_len) > 0);
      int h_row_offset = (tok_h - 1) * hidden_size;
      int tok_y = k_base + ly;
      int valid_tok_y = (tok_y < total_tokens);
      int g_row_offset = tok_y * total_gates;
#pragma unroll
      for (int w = 0; w < WPT; w++) {
        int stride = w * TS;
        // 1. Коалесцированная загрузка H_out и транспонирование на лету в
        // LeftSub[local_r][ly]
        int local_r = lx + stride;
        int global_r = group_row + local_r;
        int valid_h = valid_tok_h && (global_r < hidden_size);
        int safe_h = valid_h ? (h_row_offset + global_r) : 0;
        LeftSub[local_r][ly] = select(0.0f, H_out[safe_h], valid_h);
        // 2. Коалесцированная загрузка Gsub[ly][local_c]
        int local_c = lx + stride;
        int global_c = group_col + local_c;
        int valid_g = valid_tok_y && (global_c < total_gates);
        int safe_g = valid_g ? (g_row_offset + global_c) : 0;
        Gsub[ly][local_c] = select(0.0f, all_grad_gates[safe_g], valid_g);
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
        int offset = gr * total_gates + out_col_base;
        if (full_vec_valid) {
          vstore4(vload4(0, grad_U + offset) + rows[r], 0, grad_U + offset);
        } else {
          float vals[4] = {rows[r].x, rows[r].y, rows[r].z, rows[r].w};
          for (int c = 0; c < 4; c++) {
            if (out_col_base + c < total_gates) {
              grad_U[offset + c] += vals[c];
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
