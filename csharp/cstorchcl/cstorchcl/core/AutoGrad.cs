using System;
using System.Collections.Generic;
using cstorchcl.compiler;
using NOpenCL;

namespace cstorchcl.core
{
    /// <summary>
    /// Базовый абстрактный класс для всех математических операций в графе вычислений.
    /// Хранит контекст связей между тензорами и логику вычисления производных.
    /// </summary>
    public abstract class Op
    {
        /// <summary>
        /// Массив входных тензоров, которые участвовали в данной операции.
        /// </summary>
        public Tensor[] Inputs { get; protected set; }

        /// <summary>
        /// Выходной тензор, который был получен в результате данной операции.
        /// </summary>
        public Tensor Output { get; protected set; }

        /// <summary>
        /// Выполняет обратный проход операции: рассчитывает градиенты по входам 
        /// на основе пришедшего сверху градиента по выходу.
        /// </summary>
        /// <param name="gradOutput">Градиент функции потерь по выходу этой операции (dL/dY).</param>
        /// <param name="context">Текущий рабочий контекст OpenCL.</param>
        /// <returns>Массив градиентов по входам операции (dL/dX_i).</returns>
        public abstract Tensor[] Backward(Tensor gradOutput, Context context);
    }

    /// <summary>
    /// Статический движок автоматического дифференцирования.
    /// </summary>
    public static class AutoGrad
    {
        /// <summary>
        /// Запускает топологическую сортировку графа и выполняет обратное распространение ошибки (Backpropagation).
        /// </summary>
        /// <param name="target">Целевой тензор (обычно это скаляр Loss, от которого берется производная).</param>
        /// <param name="context">Текущий рабочий контекст OpenCL для аллокации буферов градиентов.</param>
        public static void Backward(Tensor target, Context context)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (context == null) throw new ArgumentNullException(nameof(context));

            // 1. Инициализируем градиент целевого тензора (для Loss это всегда единица: dLoss/dLoss = 1.0f)
            target.RequiresGrad = true;
            target.InitGradient(context);

            // Заполняем буфер единицей. В OpenCL это делается через очередь, 
            // но для базовой логики графа считаем, что буфер target.Grad.DataBuffer готов.
            // (В будущем вызовем activeQueue.EnqueueFillBuffer)

            // 2. Строим топологический порядок узлов графа вычислений (от выходов к входам)
            List<Tensor> topoOrder = new List<Tensor>();
            HashSet<Tensor> visited = new HashSet<Tensor>();

            void BuildTopo(Tensor node)
            {
                if (visited.Contains(node)) return;
                visited.Add(node);

                // Если тензор был создан операцией (это не лист дерева весов), идем глубже по его входам
                if (node.Creator is Op op)
                {
                    foreach (var input in op.Inputs)
                    {
                        BuildTopo(input);
                    }
                }
                topoOrder.Add(node);
            }

            BuildTopo(target);

            // Разворачиваем список, чтобы идти строго в обратном порядке (от Loss назад к весам)
            topoOrder.Reverse();

            // 3. Выполняем обход графа и аккумулируем градиенты
            foreach (var tensor in topoOrder)
            {
                if (tensor.Creator is Op op)
                {
                    // Запрашиваем у операции расчет градиентов для ее входов
                    Tensor[] inputGrads = op.Backward(tensor.Grad, context);

                    for (int i = 0; i < op.Inputs.Length; i++)
                    {
                        Tensor input = op.Inputs[i];
                        if (!input.RequiresGrad) continue;

                        // Инициализируем буфер градиента для входа, если он еще пустой
                        
                        input.InitGradient(context);

                        // Аккумулируем градиент (input.Grad += inputGrads[i])
                        // ВАЖНО: В нейросетях градиенты одного и того же тензора, используемого в разных местах, складываются.
                        // Для этого потребуется написать поэлементное суммирующее OpenCL-ядро (Add-kernel)
                        AccumulateGradient(input.Grad, inputGrads[i]);
#if DEBUG
HostReturner.PrintTensor(input, "sum_grad", true);
                        Console.WriteLine();
#endif
                        
                    }
                }
            }
        }

        /// <summary>
        /// Физически суммирует градиенты на GPU во VRAM.
        /// Реализует операцию: targetGrad += newGrad
        /// </summary>
        private static void AccumulateGradient(Tensor targetGrad, Tensor newGrad)
        {
            if (targetGrad == null || newGrad == null) return;

            string kernelSource = @"
                kernel void accum_grad(global float* target, global const float* source) {
                    int id = get_global_id(0);
                    target[id] += source[id];
                }";

            // Используем контекст и устройство из буфера тензора
            // (Так как NOpenCL.Buffer привязан к контексту, берем глобальный ComputeContext через диспетчер или activeQueue)
            var queue = DeviceQueueRegistry.GetCurrent();

            // Компилируем легкое ядро аккумуляции
            var prg = compiler.JITProgramCache.GetOrCompile(queue.Context, queue.Device, kernelSource);
            using var kernel = prg.CreateKernel("accum_grad");

            kernel.Arguments[0].SetValue(targetGrad.DataBuffer);
            kernel.Arguments[1].SetValue(newGrad.DataBuffer);

            // Запускаем параллельное сложение по всей длине буфера градиента
            queue.EnqueueNDRangeKernel(kernel,  new[] { new IntPtr(targetGrad.Size) }, null);
            queue.Finish();
        }

    }
}

