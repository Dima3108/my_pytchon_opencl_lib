// Полезные инлайн-функции активации
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
void atomic_add_float(global volatile float* source, const float val) {
    union { unsigned int intVal; float floatVal; } next, expected, current;
    current.floatVal = *source;
    do {
        expected.floatVal = current.floatVal;
        next.floatVal = expected.floatVal + val;
        current.intVal = atomic_cmpxchg((global volatile unsigned int*)source, expected.intVal, next.intVal);
    } while (current.intVal != expected.intVal);
}

/**
 * Монолитное ядро прямого прохода LSTM.
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */


kernel void lstm_forward(
    global const float* X,
    global const float* W,
    global const float* U,
    global const float* bias,
    global float* H_out,
    global float* workspace,
    global float* c_curr_buf,
    global float* h_curr_buf,
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    // get_global_id(0) отвечает за BATCH_SIZE [0 .. batch_size-1]
    // get_global_id(1) отвечает за HIDDEN_SIZE [0 .. hidden_size-1]
    int b = get_global_id(0);
    int hid = get_global_id(1);

    if (b >= batch_size || hid >= hidden_size) return;

    // Ссылки на ячейку текущего потока в глобальной памяти
    global float* c_curr_ptr = c_curr_buf + (b * hidden_size);
    global float* h_curr_ptr = h_curr_buf + (b * hidden_size);

    // Шаг 1: Инициализация (только один раз для каждого элемента скрытого состояния)
    c_curr_ptr[hid] = 0.0f;
    h_curr_ptr[hid] = 0.0f;

    // Синхронизируем инициализацию перед циклом по времени
    barrier(CLK_GLOBAL_MEM_FENCE);

    int x_offset;
    int ws_offset;
    int h_out_offset;

    // Шаг 2: Основной цикл по временной оси (T)
    for (int t = 0; t < seq_len; t++) {
        x_offset = b * (seq_len * in_features) + t * in_features;
        ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;

        // Каждый поток обсчитывает свои 4 гейта для индекса 'hid'
        for (int g = 0; g < 4; g++) {
            // Индекс гейта в плоском массиве весов: [g * hidden_size + hid]
            int gate_idx = g * hidden_size + hid;
            float val = bias[gate_idx];

            // 1. Умножение на входные веса: X_t * W
            for (int i = 0; i < in_features; i++) {
                val += X[x_offset + i] * W[i * (4 * hidden_size) + gate_idx];
            }

            // 2. Умножение на рекуррентные веса: H_{t-1} * U
            // Внимание: здесь мы последовательно читаем h_curr_ptr, 
            // который был синхронизирован барьером в конце предыдущей итерации t
            for (int h = 0; h < hidden_size; h++) {
                val += h_curr_ptr[h] * U[h * (4 * hidden_size) + gate_idx];
            }

            // Запись в workspace
            workspace[ws_offset + gate_idx] = val;
        }

        // Ждем, пока ВСЕ потоки текущего батча посчитают и запишут свои гейты в workspace
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Шаг 3: Расчет активаций и обновление состояний (без цикла по h!)
        float raw_i = workspace[ws_offset + 0 * hidden_size + hid];
        float raw_f = workspace[ws_offset + 1 * hidden_size + hid];
        float raw_g = workspace[ws_offset + 2 * hidden_size + hid];
        float raw_o = workspace[ws_offset + 3 * hidden_size + hid];

        float i_gate = sigmoid(raw_i);
        float f_gate = sigmoid(raw_f);
        float g_gate = tanh(raw_g);
        float o_gate = sigmoid(raw_o);

        // Обновляем память ячейки c_t и скрытого состояния h_t
        float c_val = f_gate * c_curr_ptr[hid] + i_gate * g_gate;
        c_curr_ptr[hid] = c_val;

        float h_val = o_gate * tanh(c_val);
        h_curr_ptr[hid] = h_val;

        // Пишем в глобальный выходной тензор
        H_out[h_out_offset + hid] = h_val;

        // ВАЖНО: Барьер в конце шага. Гарантирует, что на следующей итерации t+1
        // при чтении `h_curr_ptr[h]` все потоки увидят обновленные h_val.
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


/**
 * Монолитное ядро обратного прохода LSTM (BPTT).
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */


kernel void lstm_backward(
    global const float* X,
    global const float* W,
    global const float* U,
    global const float* H_out,
    global const float* workspace,
    global const float* grad_H_out,
    global float* grad_X,
    global float* grad_W,
    global float* grad_U,
    global float* dh_next_buf,
    global float* dc_next_buf,
    global float* c_accum_buf,
    global float* grad_gates_buf, 
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    int b = get_global_id(0);
    int hid = get_global_id(1);

    if (b >= batch_size || hid >= hidden_size) return;

    global float* dh_next = dh_next_buf + (b * hidden_size);
    global float* dc_next = dc_next_buf + (b * hidden_size);
    global float* c_accum = c_accum_buf + (b * hidden_size);
    global float* grad_gates = grad_gates_buf + (b * 4 * hidden_size);

    // Инициализация градиентов "из будущего"
    dh_next[hid] = 0.0f;
    dc_next[hid] = 0.0f;
    c_accum[hid] = 0.0f; 

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Шаг 1: Прямая регенерация цепочки состояний C (параллельно по hid)
    for (int t = 0; t < seq_len; t++) {
        int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + hid]);
        float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + hid]);
        float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + hid]);
        
        c_accum[hid] = f_gate * c_accum[hid] + i_gate * g_gate;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Шаг 2: Обратный проход во времени (BPTT)
    for (int t = seq_len - 1; t >= 0; t--) {
        int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        int h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;
        int x_offset = b * (seq_len * in_features) + t * in_features;

        float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + hid]);
        float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + hid]);
        float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + hid]);
        
        // Восстановление c_{t-1}
        float c_prev_val = (f_gate != 0.0f) ? (c_accum[hid] - i_gate * g_gate) / f_gate : 0.0f;

        float dh_curr = grad_H_out[h_out_offset + hid] + dh_next[hid];
        float raw_o = workspace[ws_offset + 3 * hidden_size + hid];
        float o_gate = sigmoid(raw_o);
        float tanh_c = tanh(c_accum[hid]);

        // Локальные градиенты по гейтам
        float d_raw_o = dh_curr * tanh_c * (o_gate * (1.0f - o_gate));
        float dc_curr = dh_curr * o_gate * (1.0f - tanh_c * tanh_c) + dc_next[hid];

        float d_raw_f = dc_curr * c_prev_val * (f_gate * (1.0f - f_gate));
        float d_raw_i = dc_curr * g_gate * (i_gate * (1.0f - i_gate));
        float d_raw_g = dc_curr * i_gate * (1.0f - g_gate * g_gate);

        // Пишем в глобальный буфер гейтов
        grad_gates[0 * hidden_size + hid] = d_raw_i;
        grad_gates[1 * hidden_size + hid] = d_raw_f;
        grad_gates[2 * hidden_size + hid] = d_raw_g;
        grad_gates[3 * hidden_size + hid] = d_raw_o;

        // Обновляем значения для следующего шага назад (t-1)
        dc_next[hid] = dc_curr * f_gate;
        c_accum[hid] = c_prev_val;

        // Барьер: все гейты текущего шага времени `t` должны быть полностью вычислены
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Вычисляем градиент по входу dL/dX_t (накапливаем вклад от нашего hid атомарно)
        int x_grad_offset = b * (seq_len * in_features) + t * in_features;
        for (int i = 0; i < in_features; i++) {
            float sum_x_contribution = 0.0f;
            for (int g = 0; g < 4; g++) {
                int gate_idx = g * hidden_size + hid;
                sum_x_contribution += grad_gates[gate_idx] * W[i * (4 * hidden_size) + gate_idx];
            }
            // Атомарно добавляем вклад этого скрытого элемента в общий вход
            atomic_add_float(&grad_X[x_grad_offset + i], sum_x_contribution);
        }

        // Вычисляем dh_next для шага (t-1). Каждый поток `hid` собирает данные из всего массива гейтов.
        float sum_h = 0.0f;
        for (int g = 0; g < 4; g++) {
            for (int h_idx = 0; h_idx < hidden_size; h_idx++) {
                int gate_idx = g * hidden_size + h_idx;
                // Вклад рекуррентных весов
                if (h_idx == hid) {
                     // Для оптимизации: здесь логика зависит от вашей структуры матрицы U.
                }
                // Нам нужно умножить вектор grad_gates на строку матрицы U^T.
                // grad_gates имеет размер [4 * hidden_size].
                // Поток `hid` должен собрать: sum_{g=0..3} sum_{h=0..hidden-1} grad_gates[g, h] * U[hid, g, h]
                sum_h += grad_gates[g * hidden_size + h_idx] * U[hid * (4 * hidden_size) + (g * hidden_size + h_idx)];
            }
        }
        
        // Синхронизируем вычисление sum_h перед записью в dh_next
        barrier(CLK_GLOBAL_MEM_FENCE);
        dh_next[hid] = sum_h;

        // НАКОПЛЕНИЕ ГРАДИЕНТОВ ПО ВЕСАМ (W, U) с использованием атомиков
        int h_prev_offset = (t > 0) ? (b * (seq_len * hidden_size) + (t - 1) * hidden_size) : -1;
        
        for (int g = 0; g < 4; g++) {
            int gate_idx = g * hidden_size + hid;
            float g_grad = grad_gates[gate_idx];

            // Градиент весов W
            for (int i = 0; i < in_features; i++) {
                float w_delta = X[x_offset + i] * g_grad;
                atomic_add_float(&grad_W[i * (4 * hidden_size) + gate_idx], w_delta);
            }

            // Градиент весов U
            if (h_prev_offset != -1) {
                for (int h = 0; h < hidden_size; h++) {
                    float u_delta = H_out[h_prev_offset + h] * g_grad;
                    atomic_add_float(&grad_U[h * (4 * hidden_size) + gate_idx], u_delta);
                }
            }
        }

        // Финальный барьер шага `t`, чтобы подготовить `dh_next` и `c_accum` к шагу `t-1`
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


/**
 * Монолитное ядро обратного прохода Stateful LSTM.
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */
kernel void lstm_hid_backward(
    global const float* X,              // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float* W,              // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float* U,              // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float* H_prev,         // Исходное H_prev из прямого прохода
    global const float* C_prev,         // Исходное C_prev из прямого прохода
    global const float* H_out,          // Вычисленный Forward выход H [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global const float* workspace,      // Сырые гейты из Forward [SEQ_LEN, BATCH_SIZE, 4 * HIDDEN_SIZE]
    global const float* grad_H_out,     // Градиент, пришедший сверху dL/dH [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float* grad_X,               // Выходной градиент по входу dL/dX [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global float* grad_H_prev,          // Выходной градиент по входному скрытому состоянию dL/dH_prev [BATCH_SIZE, HIDDEN_SIZE]
    global float* grad_C_prev,          // Выходной градиент по входному состоянию ячейки dL/dC_prev [BATCH_SIZE, HIDDEN_SIZE]
    global float* grad_W,               // Атомарный буфер градиентов весов dL/dW [IN_FEATURES, 4 * HIDDEN_SIZE]
    global float* grad_U,               // Атомарный буфер градиентов весов dL/dU [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global float* grad_bias,            // Атомарный буфер градиентов смещения dL/dB [4 * HIDDEN_SIZE]
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;

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
            
            // Если t == 0, то предыдущим состоянием ячейки является внешне переданный C_prev
            c_prev_t[h] = (t > 0) ? (f_gate != 0.0f ? (c_accum[h] - i_gate * g_gate) / f_gate : 0.0f) : C_prev[state_offset + h];
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

        // Аккумуляция весов W, U, Bias через атомики (код вырезан для краткости, идентичен базовому)
        // ...

        for(int h = 0; h < hidden_size; h++) {
            c_accum[h] = c_prev_t[h];
        }
    }

    // ВАЖНО: То, что осталось в dh_next и dc_next после цикла по времени — это 
    // и есть градиенты, которые перетекают обратно в начальные состояния чанка h_prev и c_prev!
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
    global const float* X,          // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float* W,          // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float* U,          // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float* bias,       // Смещения [4 * HIDDEN_SIZE]
    global const float* H_prev,     // Входное скрытое состояние из прошлого чанка [BATCH_SIZE, HIDDEN_SIZE]
    global const float* C_prev,     // Входное состояние ячейки из прошлого чанка [BATCH_SIZE, HIDDEN_SIZE]
    global float* H_out,            // Выходные скрытые состояния [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float* workspace,        // Буфер гейтов для автодифференцирования [SEQ_LEN, BATCH_SIZE, 4 * HIDDEN_SIZE]
    global float* final_H,          // Последнее вычисленное H (для передачи в след. чанк) [BATCH_SIZE, HIDDEN_SIZE]
    global float* final_C,          // Последнее вычисленное C (для передачи в след. чанк) [BATCH_SIZE, HIDDEN_SIZE]
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;

    // Локальные регистровые массивы потока под текущее состояние
    // Ограничим статическим размером (например, до 1024), чтобы компилятор не ругался на динамические массивы в приватной памяти
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
    global const float* grad_sliced,     // Усеченный градиент сверху [BATCH_SIZE, OLD_SEQ_LEN - 1, VOCAB_SIZE]
    global float* grad_original,         // Результирующий широкий градиент [BATCH_SIZE, OLD_SEQ_LEN, VOCAB_SIZE]
    const int old_seq_len,
    const int vocab_size
) {
    int b = get_global_id(0); // Индекс батча
    int t = get_global_id(1); // Индекс временного шага в оригинальной геометрии (0 ... OLD_SEQ_LEN - 1)

    int new_seq_len = old_seq_len - 1;

    // Вычисляем базовое смещение строки для оригинального (широкого) градиента
    long dest_row_offset = (long)(b * old_seq_len + t) * vocab_size;

    // Если это самый последний временной шаг, ошибка в него не протекала -> жестко зануляем всю строку
    if (t == old_seq_len - 1) {
        for (int v = 0; v < vocab_size; v++) {
            grad_original[dest_row_offset + v] = 0.0f;
        }
    } else {
        // Для всех остальных шагов вычисляем смещение в усеченной матрице градиентов
        long src_row_offset = (long)(b * new_seq_len + t) * vocab_size;

        // Копируем значения один к одному
        for (int v = 0; v < vocab_size; v++) {
            grad_original[dest_row_offset + v] = grad_sliced[src_row_offset + v];
        }
    }
}
