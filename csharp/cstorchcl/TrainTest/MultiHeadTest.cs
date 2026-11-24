using System;
using System.Collections.Generic;
using System.Text;

namespace TrainTest
{
    using cstorchcl.compiler;
    using cstorchcl.core;
    using global::cstorchcl.compiler;
    using global::cstorchcl.core;
    using global::cstorchcl.layers;
    using NOpenCL;
    using System;
    using System.Diagnostics;
    using System.Linq;

  
    
     public   class MultiHead
        {
         public   static void _Main()
            {
                Console.WriteLine("=== ТЕСТИРОВАНИЕ ОБУЧАЕМОГО СЛОЯ MULTI-HEAD FLASH ATTENTION ===");

                try
                {
                    // 1. Инициализация OpenCL
                    using var computeContext = new ComputeContext(preferAccelerator: true);
                    using var queue = computeContext.Context.CreateCommandQueue(computeContext.Device);
                    DeviceQueueRegistry.RegisterCurrent(queue);
                    Console.WriteLine($"Выбран девайс: {computeContext.Device.Name}");

                    // Геометрия тензора: [Batch, SeqLen, D_model]
                    // Выбираем размерность, где SeqLen > 16 (проверка нескольких тайлов по 16)
                    int batch = 4;
                    int seqLen = 64;
                    int numHeads = 4;
                    int headDim = 32; // D_model = 4 * 32 = 128 (укладывается в MAX_HEAD_DIM)
                    int dModel = numHeads * headDim;
                    int totalElements = batch * seqLen * dModel;

                    // Генерация синтетических входных и целевых данных
                    var rand = new Random(42);
                    float[] inputData = new float[totalElements];
                    float[] targetData = new float[totalElements];

                    for (int i = 0; i < totalElements; i++)
                    {
                        inputData[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * 0.5f;
                        // Простая целевая функция для проверки способности слоя обучаться
                        targetData[i] = (float)Math.Sin(inputData[i]);
                    }

                    using var xTensor = new Tensor(inputData, new[] { batch, seqLen, dModel }, computeContext.Context, "xTensor", MemoryFlags.ReadWrite, requiresGrad: true);
                    using var targetTensor = new Tensor(targetData, new[] { batch, seqLen, dModel }, computeContext.Context, "targetTensor", MemoryFlags.ReadWrite, requiresGrad: false);

                    var nnClGlob = new NnClGlob(computeContext);

                    // 2. Создание слоя MultiHeadAttention (с каузальной маской)
                    using var mhaLayer = new MultiHeadAttention(
                        computeContext,
                        nnClGlob,
                        dModel: dModel,
                        numHeads: numHeads,
                        isCausal: true,
                        layerId: "mha_block0");

                    var parameters = mhaLayer.GetParameters().ToList();
                    var optimizer = new SGD(parameters, learningRate: 0.005f, computeContext);

                    Console.WriteLine($"Конфигурация: Batch={batch}, SeqLen={seqLen}, D_model={dModel}, Heads={numHeads}, HeadDim={headDim}");
                    Console.WriteLine($"Всего обучаемых тензоров в слое: {parameters.Count} (Wq, Wk, Wv, Wo + biases)");
                    Console.WriteLine("\n--- Выполнение шагов обучения и замер скорости ---");

                    int epochs = 15;
                    double totalForwardMs = 0;
                    double totalBackwardMs = 0;
                    double totalStepMs = 0;
                    int measuredEpochs = 0;

                    for (int epoch = 0; epoch < epochs; epoch++)
                    {
                        Stopwatch swStep = Stopwatch.StartNew();

                    // Сбрасываем градиенты параметров и входа X
                    optimizer.ZeroGrad();
                        if (xTensor.Grad != null)
                        {
                            OpenCLBufferFiller.FillZero(xTensor.Grad.DataBuffer, (long)xTensor.Size * sizeof(float));
                        }

                        // А. Прямой проход (Q, K, V проекции -> Fused FlashAttention -> Wo проекция)
                        Stopwatch swFwd = Stopwatch.StartNew();
                        Tensor pred = mhaLayer.Forward(xTensor);
                        //queue.Finish();
                        swFwd.Stop();

                        // Б. Расчет величины ошибки
                        Tensor loss = nnClGlob.MseLoss(pred, targetTensor);
                        float[] lossCpu = HostReturner.ToCpu(loss);

                        // В. Обратный проход (Autograd: MSE -> Wo -> FlashAttentionBackward -> Wq, Wk, Wv -> X)
                        Stopwatch swBwd = Stopwatch.StartNew();
                        AutoGrad.Backward(loss, computeContext.Context);
                        //queue.Finish();
                        swBwd.Stop();

                        // Г. Проверка сцепки градиентов на первой эпохе
                        if (epoch == 0)
                        {
                            Console.WriteLine($"\n[Forward] Форма выхода: [{string.Join(", ", pred.Shape)}]");
                            Console.WriteLine("Анализ буферов градиентов после первого Backward:");
                            Console.WriteLine($" -> Градиент Wq.Weight создан: {mhaLayer.Wq.Weight.Grad != null}");
                            Console.WriteLine($" -> Градиент Wk.Weight создан: {mhaLayer.Wk.Weight.Grad != null}");
                            Console.WriteLine($" -> Градиент Wv.Weight создан: {mhaLayer.Wv.Weight.Grad != null}");
                            Console.WriteLine($" -> Градиент Wo.Weight создан: {mhaLayer.Wo.Weight.Grad != null}");
                            Console.WriteLine($" -> Градиент дотек до входа X:  {xTensor.Grad != null}");

                            if (mhaLayer.Wq.Weight.Grad == null || mhaLayer.Wk.Weight.Grad == null ||
                                mhaLayer.Wv.Weight.Grad == null || mhaLayer.Wo.Weight.Grad == null || xTensor.Grad == null)
                            {
                                Console.ForegroundColor = ConsoleColor.Red;
                                Console.WriteLine("\n[Ошибка] Градиенты оборвались внутри MultiHeadAttention!");
                                Console.ResetColor();
                                return;
                            }

                            float[] wqGrad = HostReturner.ToCpuGrad(mhaLayer.Wq.Weight);
                            float[] xGrad = HostReturner.ToCpuGrad(xTensor);
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.WriteLine($"[Успех] Градиент Wq (первые 4): {wqGrad[0]:F6}, {wqGrad[1]:F6}, {wqGrad[2]:F6}, {wqGrad[3]:F6}");
                            Console.WriteLine($"[Успех] Градиент X  (первые 4): {xGrad[0]:F6}, {xGrad[1]:F6}, {xGrad[2]:F6}, {xGrad[3]:F6}\n");
                            Console.ResetColor();
                        }

                        // Д. Физическое обновление весов
                        optimizer.Step();
                       // queue.Finish();
                        swStep.Stop();

                        // Пропускаем первые 2 эпохи как прогрев (JIT-компиляция ядер драйвером)
                        if (epoch >= 2)
                        {
                            totalForwardMs += swFwd.Elapsed.TotalMilliseconds;
                            totalBackwardMs += swBwd.Elapsed.TotalMilliseconds;
                            totalStepMs += swStep.Elapsed.TotalMilliseconds;
                            measuredEpochs++;
                        }

                        Console.WriteLine(
                            $"Эпоха {epoch + 1:D2}/{epochs} | Loss: {lossCpu[0]:F6} | " +
                            $"Fwd: {swFwd.Elapsed.TotalMilliseconds:F2} мс | " +
                            $"Bwd: {swBwd.Elapsed.TotalMilliseconds:F2} мс | " +
                            $"Итого шаг: {swStep.Elapsed.TotalMilliseconds:F2} мс");
                    }

                    // 3. Итоговая статистика производительности
                    if (measuredEpochs > 0)
                    {
                        double avgFwd = totalForwardMs / measuredEpochs;
                        double avgBwd = totalBackwardMs / measuredEpochs;
                        double avgStep = totalStepMs / measuredEpochs;
                        long tokensPerBatch = (long)batch * seqLen;
                        double tokensPerSec = tokensPerBatch / (avgStep / 1000.0);

                        Console.WriteLine("\n----------------------------------------------------------------");
                        Console.ForegroundColor = ConsoleColor.Cyan;
                        Console.WriteLine($"Среднее время Forward:  {avgFwd:F2} мс");
                        Console.WriteLine($"Среднее время Backward: {avgBwd:F2} мс");
                        Console.WriteLine($"Среднее время итерации: {avgStep:F2} мс ({1000.0 / avgStep:F1} it/s)");
                        Console.WriteLine($"Пропускная способность: {tokensPerSec:F0} токенов/сек");
                        Console.ResetColor();
                        Console.WriteLine("----------------------------------------------------------------");
                    }

                    Console.WriteLine("\n=== ТЕСТ MULTI-HEAD ATTENTION УСПЕШНО ЗАВЕРШЕН! ===");
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\n[КРИТИЧЕСКАЯ ОШИБКА]: {ex.Message}\n{ex.StackTrace}");
                    Console.ResetColor();
                }
                finally
                {
                    OpenCLBufferManager.DisposeAll();
                    HostPtrBufferManager.DisposeAll();
                    DeviceQueueRegistry.Clear();
                }

                Console.ReadLine();
            }
        }
    }


