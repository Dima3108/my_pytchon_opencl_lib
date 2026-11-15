kernel void accum_grad_fused_2(
    global float* t0, global const float* s0, const int len0,
    global float* t1, global const float* s1, const int len1
) {
    int tid = get_global_id(0);
    if (tid < len0) { t0[tid] += s0[tid]; }
    if (tid < len1) { t1[tid] += s1[tid]; }
}
