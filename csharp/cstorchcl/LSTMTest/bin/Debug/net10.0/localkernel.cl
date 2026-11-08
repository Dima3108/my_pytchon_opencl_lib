
                kernel void accum_grad(global float* target, global const float* source) {
                    int id = get_global_id(0);
                    target[id] += source[id];
                }