// =========================================================================
// 1. PATCHIFY FORWARD: [B, C, H, W] -> [B, num_patches, C * P * P]
// =========================================================================
__kernel void patchify_forward(
    __global const float* img,     // [B, C, H, W]
    __global float* patches,       // [B, grid_h * grid_w, C * P * P]
    const int B,
    const int C,
    const int H,
    const int W,
    const int P)
{
    int patch_dim = C * P * P;
    int d = get_global_id(0);        // [0 .. patch_dim - 1]
    int bp_idx = get_global_id(1);   // b * num_patches + patch_idx

    int grid_w = W / P;
    int grid_h = H / P;
    int num_patches = grid_h * grid_w;

    if (d >= patch_dim || bp_idx >= B * num_patches) return;

    int b = bp_idx / num_patches;
    int p_idx = bp_idx % num_patches;

    int py = p_idx / grid_w; // строка патча
    int px = p_idx % grid_w; // столбец патча

    // Разбираем индекс d внутри патча на (c, dy, dx)
    int pp = P * P;
    int c = d / pp;
    int rem = d % pp;
    int dy = rem / P;
    int dx = rem % P;

    int y = py * P + dy;
    int x = px * P + dx;

    int img_offset = ((b * C + c) * H + y) * W + x;
    patches[bp_idx * patch_dim + d] = img[img_offset];
}

// =========================================================================
// 2. PATCHIFY BACKWARD: [B, num_patches, C * P * P] -> [B, C, H, W]
// =========================================================================
__kernel void patchify_backward(
    __global const float* d_patches,
    __global float* d_img,
    const int B,
    const int C,
    const int H,
    const int W,
    const int P)
{
    int patch_dim = C * P * P;
    int d = get_global_id(0);
    int bp_idx = get_global_id(1);

    int grid_w = W / P;
    int grid_h = H / P;
    int num_patches = grid_h * grid_w;

    if (d >= patch_dim || bp_idx >= B * num_patches) return;

    int b = bp_idx / num_patches;
    int p_idx = bp_idx % num_patches;

    int py = p_idx / grid_w;
    int px = p_idx % grid_w;

    int pp = P * P;
    int c = d / pp;
    int rem = d % pp;
    int dy = rem / P;
    int dx = rem % P;

    int y = py * P + dy;
    int x = px * P + dx;

    int img_offset = ((b * C + c) * H + y) * W + x;
    d_img[img_offset] = d_patches[bp_idx * patch_dim + d];
}

// =========================================================================
// 3. ADD POSITIONAL EMBEDDING (Broadcast по Batch): Out[b, n, d] = X[b, n, d] + Pos[n, d]
// =========================================================================
__kernel void add_pos_embed_forward(
    __global const float* X,       // [B, N, D]
    __global const float* Pos,     // [1, N, D] (или [N, D])
    __global float* Out,           // [B, N, D]
    const int B,
    const int ND)                  // N * D
{
    int idx = get_global_id(0);
    if (idx >= B * ND) return;

    int pos_idx = idx % ND;
    Out[idx] = X[idx] + Pos[pos_idx];
}

// =========================================================================
// 4. BACKWARD POSITIONAL EMBEDDING: Суммируем dOut по оси Batch -> dPos[n, d]
// =========================================================================
__kernel void add_pos_embed_backward_pos(
    __global const float* dOut,    // [B, N, D]
    __global float* dPos,          // [1, N, D]
    const int B,
    const int ND)                  // N * D
{
    int pos_idx = get_global_id(0);
    if (pos_idx >= ND) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += dOut[b * ND + pos_idx];
    }
    dPos[pos_idx] = sum;
}
