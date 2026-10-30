using System;
using System.Collections.Generic;
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
                    }
                }
            }
        }

        /// <summary>
        /// Вспомогательный метод для суммирования градиентов на GPU.
        /// Реализует логику: targetGrad += newGrad
        /// </summary>
        private static void AccumulateGradient(Tensor targetGrad, Tensor newGrad)
        {
            // TODO: Вызов легкого Си-ядра поэлементного сложения векторов во VRAM через DeviceQueueRegistry.GetCurrent()
            // На этапе тестирования структуры графа здесь будет заглушка.
        }
    }
}

