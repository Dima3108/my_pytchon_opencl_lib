kernel void accum_grad_fused_1(
    global float* t0, global const float* s0, const int len0
) {
    int tid = get_global_id(0);
    if (tid < len0) { t0[tid] += s0[tid]; }
}
