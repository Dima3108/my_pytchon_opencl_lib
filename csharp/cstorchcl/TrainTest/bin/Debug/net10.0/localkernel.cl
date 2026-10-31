
                kernel void sgd_step(global float* W, global const float* G) {
                    int id = get_global_id(0);
                    W[id] -= 0.2 * G[id];
                }