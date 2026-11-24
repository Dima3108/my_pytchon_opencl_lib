__kernel void tensor_add_forward(
    __global const float* A,
    __global const float* B,
    __global float* Out,
    const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        Out[idx] = A[idx] + B[idx];
    }
}
