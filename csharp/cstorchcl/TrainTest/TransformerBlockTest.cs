using cstorchcl.compiler;
using cstorchcl.core;
using cstorchcl.layers;
using NOpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace TrainTest
{
    public class TransformerBlockTest
    {
        public static void _Main()
        {
            Console.WriteLine("=== БЕНЧМАРК И ТЕСТ TRANSFORMER BLOCK (CSTORCHCL / OPENCL) ===");

            //try
            {
                using var computeContext = new ComputeContext(preferAccelerator: true);
                using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
                DeviceQueueRegistry.RegisterCurrent(queue);
                Console.WriteLine($"Девайс: {computeContext.Device.Name}");

                // Идентичная с PyTorch геометрия
                int batch = 4;
                int seqLen = 64;
                int dModel = 128;
                int numHeads = 4;
                int intermediateDim = 344; // ~ 8/3 * 128 (стандарт LLaMA/Gemma)
                int totalElements = batch * seqLen * dModel;

                // Генерация данных (seed = 42)
                var rand = new Random(42);
                float[] inputData = new float[totalElements];
                float[] targetData = new float[totalElements];

                for (int i = 0; i < totalElements; i++)
                {
                    inputData[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * 0.5f;
                    targetData[i] = (float)Math.Sin(inputData[i]) + 0.5f * inputData[i];
                }

                using var xTensor = new Tensor(inputData, new[] { batch, seqLen, dModel }, computeContext.Context, "xTensor", MemoryFlags.ReadWrite, requiresGrad: true);
                using var targetTensor = new Tensor(targetData, new[] { batch, seqLen, dModel }, computeContext.Context, "targetTensor", MemoryFlags.ReadWrite, requiresGrad: false);

                var nnClGlob = new NnClGlob(computeContext);

                // Создание полного TransformerBlock (Pre-RMSNorm + FlashAttention + Residual + SwiGLU)
                var block = new TransformerBlock(
                    computeContext,
                    nnClGlob,
                    dModel: dModel,
                    numHeads: numHeads,
                    intermediateDim: intermediateDim,
                    isCausal: true,
                    layerId: "transformer_block_0");

                var parameters = block.GetParameters().ToList();
                var optimizer = new SGD(parameters, learningRate: 0.01f, computeContext);

                long totalParamsCount = parameters.Sum(p => (long)p.Size);
                Console.WriteLine($"Конфигурация: Batch={batch}, SeqLen={seqLen}, D_model={dModel}, Heads={numHeads}, FFN={intermediateDim}");
                Console.WriteLine($"Тензоров параметров: {parameters.Count} | Всего обучаемых весов: {totalParamsCount:N0}");
                Console.WriteLine("\n--- Запуск цикла обучения и замер скорости ---");

                int epochs = 100;
                double totalForwardMs = 0;
                double totalBackwardMs = 0;
                double totalStepMs = 0;
                int measuredEpochs = 0;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    Stopwatch swStep = Stopwatch.StartNew();

                    
                    /*if (xTensor.Grad != null)
                    {
                        OpenCLBufferFiller.FillZero(xTensor.Grad.DataBuffer, (long)xTensor.Size * sizeof(float));
                    }*/
                    /*if (xTensor.RequiresGrad)
                        xTensor.InitGradient(computeContext.Context);*/
optimizer.ZeroGrad();
                   // HostReturner.PrintTensor(xTensor, "x", true);
                    // 1. Forward
                    Stopwatch swFwd = Stopwatch.StartNew();
                    Tensor pred = block.Forward(xTensor);
                    queue.Finish();
                    swFwd.Stop();

                    // 2. Loss
                    Tensor loss = nnClGlob.MseLoss(pred, targetTensor);
                    float[] lossCpu = HostReturner.ToCpu(loss);

                    // 3. Backward
                    Stopwatch swBwd = Stopwatch.StartNew();
                    AutoGrad.Backward(loss, computeContext.Context);
                    queue.Finish();
                    swBwd.Stop();

                    // Проверка градиентов на 1-й эпохе
                    if (epoch == 0)
                    {
                        bool allGradsOk =
                            block.Norm1Gamma.Grad != null &&
                            block.Norm2Gamma.Grad != null &&
                            block.Attention.Wq.Weight.Grad != null &&
                            block.Attention.Wo.Weight.Grad != null &&
                            block.FeedForward.WeightGateUp.Grad != null &&
                            block.FeedForward.WDown.Weight.Grad != null &&
                            xTensor.Grad != null;

                        Console.WriteLine($"[Проверка AutoGrad] Все градиенты дошли до входов и параметров: {allGradsOk}");
                        if (!allGradsOk) return;
                    }

                    // 4. Optimizer Step
                    optimizer.Step();
                    queue.Finish();
                    swStep.Stop();

                    // Пропускаем первые 5 эпох как прогрев (JIT-компиляция OpenCL)
                    if (epoch >= 5)
                    {
                        totalForwardMs += swFwd.Elapsed.TotalMilliseconds;
                        totalBackwardMs += swBwd.Elapsed.TotalMilliseconds;
                        totalStepMs += swStep.Elapsed.TotalMilliseconds;
                        measuredEpochs++;
                    }

                    if (epoch == 0 || (epoch + 1) % 5 == 0)
                    {
                        Console.WriteLine(
                            $"Эпоха {epoch + 1:D2}/{epochs} | Loss: {lossCpu[0]:F6} | " +
                            $"Fwd: {swFwd.Elapsed.TotalMilliseconds:F2} мс | " +
                            $"Bwd: {swBwd.Elapsed.TotalMilliseconds:F2} мс | " +
                            $"Шаг: {swStep.Elapsed.TotalMilliseconds:F2} мс");
                    }
                }

                double avgFwd = totalForwardMs / measuredEpochs;
                double avgBwd = totalBackwardMs / measuredEpochs;
                double avgStep = totalStepMs / measuredEpochs;
                long tokensPerBatch = (long)batch * seqLen;
                double tokensPerSec = tokensPerBatch / (avgStep / 1000.0);

                Console.WriteLine("\n================================================================");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"[cstorchcl] Среднее время Forward:  {avgFwd:F3} мс");
                Console.WriteLine($"[cstorchcl] Среднее время Backward: {avgBwd:F3} мс");
                Console.WriteLine($"[cstorchcl] Среднее время итерации: {avgStep:F3} мс ({1000.0 / avgStep:F1} it/s)");
                Console.WriteLine($"[cstorchcl] Пропускная способность: {tokensPerSec:F0} токенов/сек");
                Console.ResetColor();
                Console.WriteLine("================================================================");
            }
            //catch (Exception ex)
            {
               /* Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[ОШИБКА]: {ex.Message}\n{ex.StackTrace}");
                Console.ResetColor();*/
            }
            //finally
            {
                OpenCLBufferManager.DisposeAll();
                HostPtrBufferManager.DisposeAll();
                DeviceQueueRegistry.Clear();
            }

            //Console.ReadLine();
        }
    }
}
