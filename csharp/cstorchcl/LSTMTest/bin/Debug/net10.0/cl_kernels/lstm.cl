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
    global const float* X,          // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float* W,          // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float* U,          // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float* bias,       // Смещения [4 * HIDDEN_SIZE]
    global float* H_out,            // Выходные скрытые состояния [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float* workspace,        // Буфер гейтов для автодифференцирования [SEQ_LEN, BATCH_SIZE, 4 * HIDDEN_SIZE]
    //*global float* final_C,          
    // Финальное состояние ячейки для сохранения (опционально) [BATCH_SIZE, HIDDEN_SIZE]
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    int b = get_global_id(0); // Индекс текущего батча
    if (b >= batch_size) return;

    // Выделяем локальные массивы (или регистры) для хранения текущего состояния (h_t и c_t)
    // Для больших hidden_size их можно передавать через local-память, 
    // но для простоты и изоляции батчей используем приватную память потока.
    // Внимание: размеры должны быть константами, поэтому предполагаем динамическое смещение в глобальной памяти,
    // либо инициализируем нулями на старте шага.
    
    // Временные буферы текущего шага для h_t и c_t (хранятся локально в потоке)
    // Если hidden_size очень большой, лучше аллоцировать их в глобальной памяти и занулить.
    float c_curr[1024]; 
    float h_curr[1024];

    // Инициализируем начальные состояния нулями (h_0 = 0, c_0 = 0)
    for (int h = 0; h < hidden_size; h++) {
        c_curr[h] = 0.0f;
        h_curr[h] = 0.0f;
    }

    // Основной цикл по временной оси (Forward по T)
    for (int t = 0; t < seq_len; t++) {
        
        // Индексы для текущего шага времени t
        int x_offset = b * (seq_len * in_features) + t * in_features;
        int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        int h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;

        // Вычисляем значения для 4-х гейтов
        // Формула: Gate_raw = X_t * W + H_{t-1} * U + Bias
        for (int g = 0; g < 4 * hidden_size; g++) {
            float val = bias[g];

            // 1. Умножение на входные веса: X_t * W
            for (int i = 0; i < in_features; i++) {
                val += X[x_offset + i] * W[i * (4 * hidden_size) + g];
            }

            // 2. Умножение на рекуррентные веса: H_{t-1} * U
            for (int h = 0; h < hidden_size; h++) {
                val += h_curr[h] * U[h * (4 * hidden_size) + g];
            }

            // Сохраняем "сырое" значение гейта во VRAM Workspace для Backward шага
            workspace[ws_offset + g] = val;
        }

        // Применяем функции активации и обновляем состояния ячейки (C) и скрытого слоя (H)
        for (int h = 0; h < hidden_size; h++) {
            // Извлекаем сырые значения гейтов по их смещениям
            float raw_i = workspace[ws_offset + 0 * hidden_size + h];
            float raw_f = workspace[ws_offset + 1 * hidden_size + h];
            float raw_g = workspace[ws_offset + 2 * hidden_size + h];
            float raw_o = workspace[ws_offset + 3 * hidden_size + h];

            // Считаем активации гейтов
            float i_gate = sigmoid(raw_i);
            float f_gate = sigmoid(raw_f);
            float g_gate =tanh(raw_g); // встроенный в OpenCL tanh
            float o_gate = sigmoid(raw_o);

            // Обновляем состояние ячейки: c_t = f_gate * c_{t-1} + i_gate * g_gate
            c_curr[h] = f_gate * c_curr[h] + i_gate * g_gate;

            // Вычисляем скрытое состояние: h_t = o_gate * tanh(c_t)
            h_curr[h] = o_gate * tanh(c_curr[h]);

            // Записываем результат h_t в глобальный выходной тензор
            H_out[h_out_offset + h] = h_curr[h];
        }
    }
    
    // Если нужно, сохраняем финальное состояние C для последующих слоев
    /*if (final_C != NULL) {
        for (int h = 0; h < hidden_size; h++) {
            final_C[b * hidden_size + h] = c_curr[h];
        }
    }*/
}
/**
 * Монолитное ядро обратного прохода LSTM (BPTT).
 * Сетка запуска (Global Work Size): [BATCH_SIZE]
 */
kernel void lstm_backward(
    global const float* X,              // Входные данные [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global const float* W,              // Веса входа [IN_FEATURES, 4 * HIDDEN_SIZE]
    global const float* U,              // Веса скрытого состояния [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    global const float* H_out,          // Вычисленный Forward выход H [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global const float* workspace,      // Сырые гейты из Forward [SEQ_LEN, BATCH_SIZE, 4 * HIDDEN_SIZE]
    global const float* grad_H_out,     // Градиент, пришедший сверху dL/dH [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
    global float* grad_X,               // Выходной градиент по входу dL/dX [BATCH_SIZE, SEQ_LEN, IN_FEATURES]
    global float* grad_W,               // Атомарный буфер градиентов весов dL/dW [IN_FEATURES, 4 * HIDDEN_SIZE]
    global float* grad_U,               // Атомарный буфер градиентов весов dL/dU [HIDDEN_SIZE, 4 * HIDDEN_SIZE]
    //global float* grad_bias,            // Атомарный буфер градиентов смещения dL/dB [4 * HIDDEN_SIZE]
    const int batch_size,
    const int seq_len,
    const int in_features,
    const int hidden_size
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;

    // Буферы для накопления градиентов «из будущего» (временной шаг t+1)
    float dh_next[1024];
    float dc_next[1024];
    
    // Буфер для восстановления c_t на основе сохраненных состояний
    float c_accum[1024];

    // Инициализируем градиенты из будущего нулями
    for (int h = 0; h < hidden_size; h++) {
        dh_next[h] = 0.0f;
        dc_next[h] = 0.0f;
        c_accum[h] = 0.0f; // Пересчитаем c_t честно, идя с шага 0 или храня промежуточные C. 
    }

    // ВАЖНО: Для точного вычисления dc нам нужно знать точное значение c_t и c_{t-1}.
    // Так как мы не сохраняли C во VRAM (чтобы экономить память), мы можем быстро регенерировать 
    // цепочку состояний C для текущего батча с 0 до T прямо в регистры.
    for (int t = 0; t < seq_len; t++) {
        int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        for (int h = 0; h < hidden_size; h++) {
            float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + h]);
            float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + h]);
            float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + h]);
            c_accum[h] = f_gate * c_accum[h] + i_gate * g_gate;
        }
    }

    // Начинаем обратный проход во времени (от T-1 до 0)
    for (int t = seq_len - 1; t >= 0; t--) {
        
        int ws_offset = t * (batch_size * 4 * hidden_size) + b * (4 * hidden_size);
        int h_out_offset = b * (seq_len * hidden_size) + t * hidden_size;
        int x_offset = b * (seq_len * in_features) + t * in_features;

        // Извлекаем "предыдущее" состояние c_{t-1}. Если t == 0, то оно равно 0.
        // Для этого откатываем c_accum назад.
        float c_prev[1024];
        for (int h = 0; h < hidden_size; h++) {
            float i_gate = sigmoid(workspace[ws_offset + 0 * hidden_size + h]);
            float f_gate = sigmoid(workspace[ws_offset + 1 * hidden_size + h]);
            float g_gate = tanh(workspace[ws_offset + 2 * hidden_size + h]);
            
            // Математически откатываем c_{t-1} = (c_t - i * g) / f
            // Во избежание деления на 0 при занулении f_gate, надежнее хранить C в workspace, 
            // но в целях экономии VRAM восстанавливаем через сохраненный forward проход, 
            // либо считываем h_{t-1}
            c_prev[h] = (f_gate != 0.0f) ? (c_accum[h] - i_gate * g_gate) / f_gate : 0.0f;
        }

        // Локальные градиенты на текущем шаге времени t
        float grad_gates[4 * 1024]; 

        for (int h = 0; h < hidden_size; h++) {
            // Текущий градиент по H состоит из пришедшего сверху (grad_H_out) + из шага (t+1) (dh_next)
            float dh_curr = grad_H_out[h_out_offset + h] + dh_next[h];

            // Извлекаем значения сырых гейтов
            float raw_i = workspace[ws_offset + 0 * hidden_size + h];
            float raw_f = workspace[ws_offset + 1 * hidden_size + h];
            float raw_g = workspace[ws_offset + 2 * hidden_size + h];
            float raw_o = workspace[ws_offset + 3 * hidden_size + h];

            float i_gate = sigmoid(raw_i);
            float f_gate = sigmoid(raw_f);
            float g_gate = tanh(raw_g);
            float o_gate = sigmoid(raw_o);

            float tanh_c = tanh(c_accum[h]);

            // 1. Градиент по Output гейту: dL/do = dh * tanh(c_t)
            // Производная сигмоиды: do/draw_o = o * (1 - o)
            float d_raw_o = dh_curr * tanh_c * (o_gate * (1.0f - o_gate));

            // 2. Градиент по состоянию ячейки: dL/dc_t = dh * o * (1 - tanh^2(c_t)) + dc_next
            float dc_curr = dh_curr * o_gate * (1.0f - tanh_c * tanh_c) + dc_next[h];

            // 3. Градиенты по остальным гейтам
            // dL/df (Forget) = dc * c_{t-1} -> df/draw_f = f * (1 - f)
            float d_raw_f = dc_curr * c_prev[h] * (f_gate * (1.0f - f_gate));

            // dL/di (Input) = dc * g_gate -> di/draw_i = i * (1 - i)
            float d_raw_i = dc_curr * g_gate * (i_gate * (1.0f - i_gate));

            // dL/dg (Candidate) = dc * i_gate -> dg/draw_g = 1 - g^2
            float d_raw_g = dc_curr * i_gate * (1.0f - g_gate * g_gate);

            // Сохраняем градиенты гейтов в локальный массив
            grad_gates[0 * hidden_size + h] = d_raw_i;
            grad_gates[1 * hidden_size + h] = d_raw_f;
            grad_gates[2 * hidden_size + h] = d_raw_g;
            grad_gates[3 * hidden_size + h] = d_raw_o;

            // Передаем градиент по C на предыдущий шаг (t-1)
            dc_next[h] = dc_curr * f_gate;
        }

        // Вычисляем градиент по входу dL/dX_t и накапливаем градиенты по рекуррентному H (dh_next)
        // dL/dX_t = grad_gates * W^T
        int x_grad_offset = b * (seq_len * in_features) + t * in_features;
        for (int i = 0; i < in_features; i++) {
            float sum_x = 0.0f;
            for (int g = 0; g < 4 * hidden_size; g++) {
                sum_x += grad_gates[g] * W[i * (4 * hidden_size) + g];
            }
            grad_X[x_grad_offset + i] = sum_x;
        }

        // Вычисляем dh_next для шага (t-1): dh_next = grad_gates * U^T
        for (int h = 0; h < hidden_size; h++) {
            float sum_h = 0.0f;
            for (int g = 0; g < 4 * hidden_size; g++) {
                sum_h += grad_gates[g] * U[h * (4 * hidden_size) + g];
            }
            dh_next[h] = sum_h;
        }

        // НАКОПЛЕНИЕ ГРАДИЕНТОВ ПО ВЕСАМ (W, U, Bias)
        // Так как разные батчи пишут в общую память весов, используем atomic_add 
        // (Требуется расширение cl_khr_global_int32_base_atomics и каст float через union, либо атомики на уровне C#)
        // Для простоты реализации на GPU часто делают редукцию по батчу, но здесь применим атомики:
        
        // Смещение для H_{t-1}
        int h_prev_offset = (t > 0) ? (b * (seq_len * hidden_size) + (t - 1) * hidden_size) : -1;

        for (int g = 0; g < 4 * hidden_size; g++) {
            float g_grad = grad_gates[g];
            
            // Атомарно прибавляем к градиенту bias
            // Примечание: в OpenCL нет стандартного atomic_add для float, 
            // на практике используют специальный трюк с atomic_cmpxchg (см. ниже)
            
            // Градиент весов W: X_t^T * grad_gates
            for (int i = 0; i < in_features; i++) {
                float w_delta = X[x_offset + i] * g_grad;
                // Обертка над атомарным сложением float:
                atomic_add_float(&grad_W[i * (4 * hidden_size) + g], w_delta);
            }

            // Градиент весов U: H_{t-1}^T * grad_gates
            if (h_prev_offset != -1) {
                for (int h = 0; h < hidden_size; h++) {
                    float u_delta = H_out[h_prev_offset + h] * g_grad;
                    // atomic_add_float(&grad_U[h * (4 * hidden_size) + g], u_delta);
                }
            }
        }

        // Шаг назад по состояниям C
        for(int h=0; h<hidden_size; h++) {
            c_accum[h] = c_prev[h];
        }
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
