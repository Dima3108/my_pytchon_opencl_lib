__kernel void tensor_copy(__global const float* src, __global float* dst, const int total_elements) {
    int tid = get_global_id(0);
    if (tid < total_elements) {
        dst[tid] = src[tid];
    }
}
