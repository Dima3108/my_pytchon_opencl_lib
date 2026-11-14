
            kernel void fill_zero_vec16(
                global uint4* buf_vec,
                global uchar* buf_bytes,
                const long num_vec16,
                const long total_bytes
            ) {
                long gid = get_global_id(0);

                // Основной проход: пишем по 16 байт (4 float / 4 int) за раз
                if (gid < num_vec16) {
                    buf_vec[gid] = (uint4)(0u, 0u, 0u, 0u);
                }

                // Хвост: если размер буфера в байтах не кратен 16
                long tail_offset = num_vec16 * 16L + gid;
                if (tail_offset < total_bytes) {
                    buf_bytes[tail_offset] = (uchar)0;
                }
            }
        