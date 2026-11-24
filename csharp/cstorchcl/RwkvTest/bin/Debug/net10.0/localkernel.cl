kernel void accum_grad_fused_4(
    global float* t0, global const float* s0, const int len0,
    global float* t1, global const float* s1, const int len1,
    global float* t2, global const float* s2, const int len2,
    global float* t3, global const float* s3, const int len3
) {
    int tid = get_global_id(0);
    if (tid < len0) { t0[tid] += s0[tid]; }
    if (tid < len1) { t1[tid] += s1[tid]; }
    if (tid < len2) { t2[tid] += s2[tid]; }
    if (tid < len3) { t3[tid] += s3[tid]; }
}
