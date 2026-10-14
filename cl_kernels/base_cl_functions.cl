// ============================================================================
// БАЗОВАЯ МАТЕМАТИКА И ТРИГОНОМЕТРИЯ
// ============================================================================

inline float nn_sin(float x) {
    return sin(x);
}

inline float nn_cos(float x) {
    return cos(x);
}

inline float nn_tanh(float x) {
    return tanh(x);
}

inline float nn_exp(float x) {
    return exp(x);
}

// Поэлементное возведение в квадрат (часто нужно для RMSNorm и дисперсии)
inline float nn_square(float x) {
    return x * x;
}

// ============================================================================
// ФУНКЦИИ АКТИВАЦИИ ДЛЯ LLM (Gemma, Qwen, Llama, GPT)
// ============================================================================

// Стандартный Sigmoid (используется в гейтах рекуррентных блоков и SSM)
inline float nn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// SiLU (Swish) — базовая активация для SwiGLU блоков (Llama, Qwen, Gemma)
// Формула: x * sigmoid(x)
inline float nn_silu(float x) {
    return x * nn_sigmoid(x);
}

// GELU (Gaussian Error Linear Unit) — используется в GPT, BERT, DeepSeek
// Аппроксимация через tanh, так как она считается на GPU значительно быстрее точного ERF
// Формула: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float nn_gelu(float x) {
    float x_cubed = x * x * x;
    float inner = 0.79788456f * (x + 0.044715f * x_cubed); // 0.79788456 = sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(inner));
}

// ReLU — классическая активация, иногда применяется в легковесных sub-блоках
inline float nn_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ============================================================================
// ГРАДИЕНТЫ БАЗОВОЙ МАТЕМАТИКИ
// ============================================================================

// d/dx [sin(x)] = cos(x)
inline float nn_sin_backward(float x, float grad_output) {
    return grad_output * cos(x);
}

// d/dx [cos(x)] = -sin(x)
inline float nn_cos_backward(float x, float grad_output) {
    return grad_output * (-sin(x));
}

// d/dx [tanh(x)] = 1 - tanh^2(x)
// ОПТИМИЗАЦИЯ: Если у вас уже есть вычисленный на forward tanh_x, лучше передавать его, чтобы не считать tanh дважды
inline float nn_tanh_backward(float x, float grad_output) {
    float t = tanh(x);
    return grad_output * (1.0f - t * t);
   //расчет по y
   //return grad_output * (1.0f - x * x);
}

// d/dx [x^2] = 2 * x
inline float nn_square_backward(float x, float grad_output) {
    return grad_output * 2.0f * x;
}

// ============================================================================
// ГРАДИЕНТЫ ФУНКЦИЙ АКТИВАЦИИ
// ============================================================================

// d/dx [sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
inline float nn_sigmoid_backward(float x, float grad_output) {
float s = nn_sigmoid(x);
    return grad_output * s * (1.0f - s); 
  

}

// d/dx [silu(x)] = silu(x) + sigmoid(x) * (1 - silu(x))
// Альтернативная форма: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
inline float nn_silu_backward(float x, float grad_output) {
    float sig = nn_sigmoid(x);
    return grad_output * (sig * (1.0f + x * (1.0f - sig)));
}

// d/dx [gelu(x)] — производная аппроксимации GELU
inline float nn_gelu_backward(float x, float grad_output) {
    float x_sq = x * x;
    float x_cubed = x_sq * x;
    float inner = 0.79788456f * (x + 0.044715f * x_cubed);
    float tanh_inner = tanh(inner);
    
    // Производная внутренней части tanh
    float d_inner = 0.79788456f * (1.0f + 3.0f * 0.044715f * x_sq);
    // Производная самого tanh
    float sech_sq = 1.0f - tanh_inner * tanh_inner;
    
    return grad_output * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_sq * d_inner);
}

// d/dx [relu(x)] = x > 0 ? 1 : 0
inline float nn_relu_backward(float x, float grad_output) {
    return x > 0.0f ? grad_output : 0.0f;
}
//расчет по y
//inline float nn_tanh_backward(float x, float grad_output) { return grad_output * (1.0f - x * x); }

// ============================================================================
// ОБРАТНЫЙ ПРОХОД (BACKWARD) — СЧИТАЕМ СТРОГО ПО ВЫХОДНОМУ ЗНАЧЕНИЮ (Y)
// ============================================================================

// Вместо: go * sigmoid(x) * (1 - sigmoid(x))
// Пишем экономичную формулу, где y — это уже вычисленный gate_A[tid] (результат сигмоиды)
inline float nn_sigmoid_backward_by_y(float y, float grad_output) { 
    return grad_output * y * (1.0f - y); 
}

// Вместо: go * (1 - tanh^2(x))
// Пишем формулу, где y — это результат тангенса
inline float nn_tanh_backward_by_y(float y, float grad_output) { 
    return grad_output * (1.0f - y * y); 
}