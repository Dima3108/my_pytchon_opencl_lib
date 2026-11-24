kernel void cat_2tensors_forward(
    global const float* X1,
    global const float* X2,
    global float* Y,
    const int num_rows,
    const int dim1,
    const int dim2
) {
    int col = get_global_id(0); // [0 .. max(dim1, dim2) - 1] -> быстрая ось
    int row = get_global_id(1); // [0 .. num_rows - 1]

    if (row >= num_rows) return;

    int out_dim = dim1 + dim2;
    int dst_row_offset = row * out_dim;

    if (col < dim1) {
        Y[dst_row_offset + col] = X1[row * dim1 + col];
    }
    if (col < dim2) {
        Y[dst_row_offset + dim1 + col] = X2[row * dim2 + col];
    }
}

kernel void cat_2tensors_backward(
    global const float* grad_output,
    global float* grad_input1,
    global float* grad_input2,
    const int num_rows,
    const int dim1,
    const int dim2
) {
    int col = get_global_id(0); // [0 .. max(dim1, dim2) - 1]
    int row = get_global_id(1); // [0 .. num_rows - 1]

    if (row >= num_rows) return;

    int out_dim = dim1 + dim2;
    int src_row_offset = row * out_dim;

    if (col < dim1) {
        grad_input1[row * dim1 + col] = grad_output[src_row_offset + col];
    }
    if (col < dim2) {
        grad_input2[row * dim2 + col] = grad_output[src_row_offset + dim1 + col];
    }
}

