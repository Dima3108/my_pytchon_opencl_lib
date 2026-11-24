using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace WkvCpuBenchmark
{
    internal class Program
    {
        
        static void Main()
        {
            // Типичные параметры (например, для модели ~1.5B - 3B или среднего слоя)
            int B = 4;      // Batch size
            int T = 512;    // Sequence length
            int C = 1024;   // Channels (должно быть кратно Vector<float>.Count, обычно 8 или 16)
            int iterations = 100;

            Console.WriteLine($"SIMD Vector<float> width: {Vector<float>.Count} floats ({Vector<float>.Count * 32}-bit)");
            Console.WriteLine($"Shape: B={B}, T={T}, C={C} | Iterations: {iterations}\n");

            var rng = new Random(42);
            float[] w = new float[C];
            float[] u = new float[C];
            float[] k = new float[B * T * C];
            float[] v = new float[B * T * C];
            float[] y = new float[B * T * C];

            for (int c = 0; c < C; c++)
            {
                w[c] = -(float)Math.Exp(rng.NextDouble() * 2.0 - 1.0); // Отрицательное затухание
                u[c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            for (int i = 0; i < k.Length; i++)
            {
                k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            // Прогрев JIT (Warmup)
            for (int i = 0; i < 10; i++)
            {
                WkvForwardSimd(w, u, k, v, y, B, T, C);
            }

            // Замер 100 итераций
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                WkvForwardSimd(w, u, k, v, y, B, T, C);
            }
            sw.Stop();

            double totalSec = sw.Elapsed.TotalSeconds;
            double itPerSec = iterations / totalSec;
            Console.WriteLine($"[C# SIMD + Parallel] Total time: {sw.Elapsed.TotalMilliseconds:F2} ms");
            Console.WriteLine($"[C# SIMD + Parallel] Speed:      {itPerSec:F2} it/s");
            Console.WriteLine($"Check y[0]: {y[0]:F6}, y[last]: {y[^1]:F6}");
        }

        static void WkvForwardSimd(float[] w, float[] u, float[] k, float[] v, float[] y, int B, int T, int C)
        {
            int vecSize = Vector<float>.Count;
            var negInf = new Vector<float>(-1e38f);

            // Параллелим по батчам (и при желании можно по блокам каналов)
            Parallel.For(0, B, b =>
            {
                int batchOffset = b * T * C;

                // Векторный проход по каналам C
                for (int c = 0; c <= C - vecSize; c += vecSize)
                {
                    var vw = new Vector<float>(w, c);
                    var vu = new Vector<float>(u, c);

                    var aa = Vector<float>.Zero;
                    var bb = Vector<float>.Zero;
                    var pp = negInf;

                    for (int t = 0; t < T; t++)
                    {
                        int offset = batchOffset + t * C + c;
                        var kk = new Vector<float>(k, offset);
                        var vv = new Vector<float>(v, offset);

                        // 1. Выход y_t с учетом бонуса u
                        var ww = vu + kk;
                        var p = Vector.Max(pp, ww);
                        var e1 = FastExpNegative(pp - p);
                        var e2 = FastExpNegative(ww - p);

                        var res = (e1 * aa + e2 * vv) / (e1 * bb + e2);
                        res.CopyTo(y, offset);

                        // 2. Обновление рекуррентного состояния с затуханием w
                        ww = vw + pp;
                        p = Vector.Max(ww, kk);
                        e1 = FastExpNegative(ww - p);
                        e2 = FastExpNegative(kk - p);

                        aa = e1 * aa + e2 * vv;
                        bb = e1 * bb + e2;
                        pp = p;
                    }
                }
            });
        }

        /// <summary>
        /// Быстрая SIMD-экспонента для неположительных аргументов (x <= 0).
        /// Использует разложение x = n * ln(2) + r и полином 4-й степени на [-ln(2), 0].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static Vector<float> FastExpNegative(Vector<float> x)
        {
            // Защита от underflow (-87f -> exp(-87) ~ 0)
            x = Vector.Max(x, new Vector<float>(-87.0f));

            var invLn2 = new Vector<float>(1.4426950408889634f);
            var ln2Hi = new Vector<float>(0.693359375f);
            var ln2Lo = new Vector<float>(-2.12194440e-4f);

            // n = round(x / ln(2))
            var fx = x * invLn2;
            var n = Vector.ConvertToInt32(fx - new Vector<float>(0.5f)); // floor для отрицательных
            var fn = Vector.ConvertToSingle(n);

            // r = x - n * ln(2)
            var r = (x - fn * ln2Hi) - fn * ln2Lo;

            // Полином Тейлора / Ремеза для exp(r)
            var p = new Vector<float>(0.041666668f); // 1/24
            p = p * r + new Vector<float>(0.16666667f); // 1/6
            p = p * r + new Vector<float>(0.5f);        // 1/2
            p = p * r + Vector<float>.One;
            p = p * r + Vector<float>.One;

            // Умножаем на 2^n через сдвиг экспоненты IEEE 754 float
            var pow2n = Vector.AsVectorSingle((n + new Vector<int>(127)) << 23);
            return p * pow2n;
        }
    }
}
