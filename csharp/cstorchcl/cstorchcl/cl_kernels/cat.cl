// Ядро прямого прохода: Склеивает два тензора по последней оси
kernel void cat_2tensors_forward(
    global const float* X1,
    global const float* X2,
    global float* Y,
    const int num_rows,
    const int dim1,
    const int dim2)
{
    int row = get_global_id(0);
    if (row >= num_rows) return;

    int out_dim = dim1 + dim2;
    int src_row_offset_x1 = row * dim1;
    int src_row_offset_x2 = row * dim2;
    int dst_row_offset_y  = row * out_dim;

    // Копируем первую часть (X1)
    for (int i = 0; i < dim1; i++) {
        Y[dst_row_offset_y + i] = X1[src_row_offset_x1 + i];
    }
    // Копируем вторую часть (X2)
    for (int i = 0; i < dim2; i++) {
        Y[dst_row_offset_y + dim1 + i] = X2[src_row_offset_x2 + i];
    }
}

// Ядро обратного прохода: Разрезает градиент обратно по двум исходным буферам
kernel void cat_2tensors_backward(
    global const float* grad_output,
    global float* grad_input1,
    global float* grad_input2,
    const int num_rows,
    const int dim1,
    const int dim2)
{
    int row = get_global_id(0);
    if (row >= num_rows) return;

    int out_dim = dim1 + dim2;
    int dst_row_offset_x1 = row * dim1;
    int dst_row_offset_x2 = row * dim2;
    int src_row_offset_y  = row * out_dim;

    // Раздаем градиент в первый тензор
    for (int i = 0; i < dim1; i++) {
        grad_input1[dst_row_offset_x1 + i] = grad_output[src_row_offset_y + i];
    }
    // Раздаем градиент во второй тензор
    for (int i = 0; i < dim2; i++) {
        grad_input2[dst_row_offset_x2 + i] = grad_output[src_row_offset_y + dim1 + i];
    }
}
