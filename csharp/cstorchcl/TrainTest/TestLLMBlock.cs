using System;
using NOpenCL;
using cstorchcl.compiler;
using cstorchcl.core;

public class TestLLMBlock : NeuralNetwork
{
    // Обучаемые параметры модели
    public Tensor RmsGamma { get; private set; }
    public Tensor ProjectionWeights { get; private set; }

    // Хранилище обычных CPU-данных, извлеченных из GPU
    public float[] ProjectedHost { get; private set; }

    public TestLLMBlock(ComputeContext computeContext, NnClGlob nnClGlob, int dModel)
        : base(computeContext, nnClGlob)
    {
        // 1. Аллокация памяти во VRAM девайса
        RmsGamma = RegisterParameter(new[] { dModel });
        ProjectionWeights = RegisterParameter(new[] { 1, dModel, dModel }); // Форма для BMM: [Batch, K, N]

        // Получаем текущую активную очередь команд для инициализации
        CommandQueue queue = DeviceQueueRegistry.GetCurrent();

        // =========================================================================
        // ИСПРАВЛЕНИЕ: Физическое заполнение параметров дефолтными весами во VRAM
        // =========================================================================

        // А. Инициализация весов RMSNorm (заполняем единицами, как в PyTorch по умолчанию)
        float[] hostGammaData = new float[RmsGamma.Size];
        Array.Fill(hostGammaData, 1.0f);

        long gammaSizeInBytes = (long)RmsGamma.Size * sizeof(float);
        unsafe
        {
            fixed (float* hgd = hostGammaData)
            {
  queue.EnqueueWriteBuffer(RmsGamma.DataBuffer, true, 0, gammaSizeInBytes, (IntPtr)hgd);
            }
        }
      

        // Б. Инициализация весов ProjectionWeights (заполняем небольшими случайными/дефолтными весами)
        // Чтобы при BMM значения не улетали в бесконечность, заполним их, например, 0.05f
        float[] hostWeightsData = new float[ProjectionWeights.Size];
        Array.Fill(hostWeightsData, 0.05f);

        long weightsSizeInBytes = (long)ProjectionWeights.Size * sizeof(float);
        unsafe
        {
            fixed (float* hwhd = hostWeightsData)
            {
queue.EnqueueWriteBuffer(ProjectionWeights.DataBuffer, true, 0, weightsSizeInBytes, (IntPtr)hwhd);
            }
        }
        

        // Гарантируем, что видеокарта завершила запись весов перед тем, как модель пойдет в работу
        queue.Finish();
        // =========================================================================
    }

    public Tensor Forward(Tensor x)
    {
        Tensor normalized = NnClGlob.RmsNorm(x, RmsGamma);
        Tensor projected = NnClGlob.Bmm(normalized, ProjectionWeights);

        ProjectedHost = HostReturner.ToCpu(projected);

#if DEBUG
        // Добавляем нелинейность ELU прямо в базовый проход
        Tensor activated = NnClGlob.Elu(projected, alpha: 1.0f);
#else
        Tensor activated = projected;
#endif

        Tensor probabilities = NnClGlob.Softmax(activated, dim: -1);
        return probabilities;
    }

}


