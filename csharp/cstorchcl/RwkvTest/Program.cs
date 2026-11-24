using cstorchcl.compiler;
using cstorchcl.core;
using cstorchcl.layers;
using cstorchcl.optimizers;
using System.Collections;

namespace RwkvTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ComputeContext computeContext = new ComputeContext();
            DeviceQueueRegistry.RegisterCurrent(computeContext.Context.CreateCommandQueue(computeContext.Device));
            NnClGlob clglob = new NnClGlob(computeContext);
            RunRwkvTrainabilityTest(computeContext,clglob);
        }
        public static void RunRwkvTrainabilityTest(ComputeContext computeCtx, NnClGlob glob)
        {
            int B = 2;
            int T = 16;
            int C = 32;
            int dFfn = C * 3;
            var rng = new Random(123);

            // 1. Генерируем синтетическую пару X и Y с временной зависимостью
            float[] xData = new float[B * T * C];
            float[] yData = new float[B * T * C];

            for (int i = 0; i < xData.Length; i++)
                xData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            for (int b = 0; b < B; b++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        int idx = b * T * C + t * C + c;
                        float curr = xData[idx];
                        float prev1 = (t >= 1) ? xData[b * T * C + (t - 1) * C + (c + 1) % C] : 0.0f;
                        float prev2 = (t >= 2) ? xData[b * T * C + (t - 2) * C + c] : 0.0f;

                        // Нелинейная функция с памятью на 2 шага назад
                        yData[idx] = (float)Math.Tanh(curr * 0.7f + prev1 * 0.5f - prev2 * 0.4f);
                    }
                }
            }

            Tensor x = new Tensor(xData, new[] { B, T, C }, computeCtx.Context, "test_x", requiresGrad: false);
            Tensor targetY = new Tensor(yData, new[] { B, T, C }, computeCtx.Context, "test_y", requiresGrad: false);

            // 2. Инициализируем блок RWKV и оптимизатор Adam
            var rwkv = new RwkvBlock(C, dFfn, computeCtx, glob, "rwkv_block_0", rng);
            var optimizer = new Adam(computeCtx, rwkv.Parameters(), lr: 0.005f);

            Console.WriteLine("=== Старт теста обучаемости RWKV Block ===");

            for (int epoch = 1; epoch <= 200; epoch++)
            {
                optimizer.ZeroGrad();

                // Forward
                Tensor predY = rwkv.Forward(x);

                // MSE Loss (используем ваш MseLossOp)
                Tensor loss = glob.MseLoss(predY, targetY);
              
                // Backward
                AutoGrad.Backward(loss, computeCtx.Context);
                /*if ((epoch+1) %9 ==0)
                {
                    HostReturner.PrintTensor(loss,"lossTensor", true);
                    foreach(var par in rwkv.Parameters())
                    {
                        HostReturner.PrintTensor(par, par.TensorID, true);
                    }
                }*/
                // Шаг оптимизации
                optimizer.Step();

                if (epoch == 1 || epoch % 20 == 0)
                {
                    // Читаем скалярное значение loss с GPU на CPU для логирования
                    float[] lossHost = HostReturner.ToCpu(loss);
                    Console.WriteLine($"Epoch {epoch,3} | MSE Loss: {lossHost[0]:F6}");
                }
            }
        }

    }
}
