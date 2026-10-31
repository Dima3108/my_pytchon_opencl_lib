using System;
using System.Collections.Generic;
using NOpenCL;
using cstorchcl.compiler;
using cstorchcl.core;

public class TestLLMBlock : NeuralNetwork
{
    // Обучаемые параметры модели
    public Tensor RmsGamma { get; private set; }
    public Tensor ProjectionWeights { get; private set; }

    // Поля для отладки прямого прохода на CPU (хосте) через HostReturner
    public float[] NormalizedHost { get; private set; }
    public float[] ProjectedHost { get; private set; }
    public float[] ActivatedHost { get; private set; }

    public TestLLMBlock(ComputeContext computeContext, NnClGlob nnClGlob, int dModel)
        : base(computeContext, nnClGlob)
    {
        // 1. Выделяем память во VRAM
        RmsGamma = RegisterParameter(new[] { dModel });
        ProjectionWeights = RegisterParameter(new[] { 1, dModel, dModel }); // [Batch, K, N]

        // 2. Чистая инициализация весов (без мусора VRAM)
        CommandQueue queue = DeviceQueueRegistry.GetCurrent();

        float[] hostGammaData = new float[RmsGamma.Size];
        Array.Fill(hostGammaData, 1.0f); // One-init для нормализации
        unsafe
        {
            fixed (float* hgd = hostGammaData)
            {
  queue.EnqueueWriteBuffer(RmsGamma.DataBuffer, true, 0, (long)RmsGamma.Size * sizeof(float), (IntPtr)hgd);
            }
        }
      

        float[] hostWeightsData = new float[ProjectionWeights.Size];
        Array.Fill(hostWeightsData, 0.05f); // Инициализируем небольшими дефолтными значениями
        unsafe
        {
            fixed (float* hwd = hostWeightsData)
            {
queue.EnqueueWriteBuffer(ProjectionWeights.DataBuffer, true, 0, (long)ProjectionWeights.Size * sizeof(float), (IntPtr)hwd);
            }
        }
        

        queue.Finish();
    }

    /// <summary>
    /// Полный сквозной Forward Pass модели: RMSNorm -> BMM -> LocalJIT (Tanh/Div/Dropout) -> Softmax
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        // 1. Глобальный слой нормализации
        Tensor normalized = NnClGlob.RmsNorm(x, RmsGamma);
        NormalizedHost = HostReturner.ToCpu(normalized); // Отладочный перехват

        // 2. Линейная проекция через пакетное умножение матриц
        Tensor projected = NnClGlob.Bmm(normalized, ProjectionWeights);
        ProjectedHost = HostReturner.ToCpu(projected); // Отладочный перехват

        // 3. Локальный JIT-блок активаций и регуляризации на регистрах GPU (Сцепка)
        var tensorArgs = new Dictionary<string, Tensor> { { "P", projected } };
        var constants = new Dictionary<string, float> { { "q", 2.0f } };
        string pseudoCode = @"
            P = nnclloc.tanh(P);
            P = P / q;
            P = nnclloc.dropout(P, 0.1, 1337);
        ";

        // Вызываем RunLocal, встроенный в подсистему модели (NnClLoc.Run)
        Dictionary<string, Tensor> localOutputs = NnClGlob.RunLocal(pseudoCode, tensorArgs, constants);
        Tensor activated = localOutputs["P"];
        ActivatedHost = HostReturner.ToCpu(activated); // Отладочный перехват

        // 4. Финальный глобальный Softmax
        Tensor probabilities = NnClGlob.Softmax(activated, dim: -1); //

        return probabilities;
    }
}
