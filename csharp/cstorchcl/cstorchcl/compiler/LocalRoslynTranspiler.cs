using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace cstorchcl.compiler
{
    public class LocalRoslynTranspiler : CSharpSyntaxRewriter
    {
        private readonly StringBuilder _bodyCode = new StringBuilder();
        private readonly StringBuilder _backwardBodyCode = new StringBuilder();

        public HashSet<string> DiscoveredTensors { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private Dictionary<string, float> _externalConstants;

        // Словарь для быстрого доступа к метаданным тензоров во время обхода AST
        private Dictionary<string, core.Tensor> _tensorMeta;

        public (string forwardSource, string backwardSource) TranslateWithGrad(
            string pseudoCode,
            Dictionary<string, float> constants,
            Dictionary<string, core.Tensor> tensors)
        {
            _externalConstants = constants ?? new Dictionary<string, float>();
            _tensorMeta = tensors ?? throw new ArgumentNullException(nameof(tensors));

            _bodyCode.Clear();
            _backwardBodyCode.Clear();
            DiscoveredTensors.Clear();
            OutputTensorNames.Clear(); // Очищаем список выходов перед новым проходом
            SyntaxTree tree = CSharpSyntaxTree.ParseText(pseudoCode);
            CompilationUnitSyntax root = tree.GetCompilationUnitRoot();
            Visit(root);

            string utilities = @"// Встроенные утилиты генерации шума и Dropout
inline float nn_rand(int tid, unsigned int seed) {
    unsigned int x = tid + seed;
    x = (x ^ 61) ^ (x >> 16);
    x += (x << 3);
    x = x ^ (x >> 4);
    x *= 0x27d4eb2d;
    x = x ^ (x >> 15);
    return (float)x / 4294967296.0f; 
}
inline float nn_dropout(float x, float p, int tid, unsigned int seed) {
    if (p <= 0.0f) return x;
    if (p >= 1.0f) return 0.0f;
    float r = nn_rand(tid, seed);
    return (r < p) ? 0.0f : x / (1.0f - p);
}
inline float nn_dropout_backward(float grad_out, float p, int tid, unsigned int seed) {
    float r = nn_rand(tid, seed);
    return (r < p) ? 0.0f : grad_out / (1.0f - p);
}

";

            // Вычисляем максимальный ранг (количество осей) среди всех обнаруженных тензоров
            var activeTensors = DiscoveredTensors.Select(name => _tensorMeta[name]).ToList();
            int maxRank = activeTensors.Max(t => t.Shape.Length);

            // За эталонную сетку берем форму самого большого тензора (у которого ранг максимален)
            var maxTensor = activeTensors.First(t => t.Shape.Length == maxRank);
            int[] maxShape = maxTensor.Shape;

            // --- ГЕНЕРАЦИЯ ДЕКОДЕРА КООРДИНАТ (Broadcasting Core) ---
            var coordDecoder = new StringBuilder();
            coordDecoder.Append("    // Восстановление многомерных координат из плоского tid\n");
            coordDecoder.Append($"    int rem = tid;\n");

            // Генерируем переменные координат: dim_0, dim_1...
            for (int i = maxRank - 1; i >= 0; i--)
            {
                if (i > 0)
                {
                    coordDecoder.Append($"    int dim_{i} = rem % {maxShape[i]};\n");
                    coordDecoder.Append($"    rem /= {maxShape[i]};\n");
                }
                else
                {
                    coordDecoder.Append($"    int dim_{i} = rem;\n");
                }
            }
            coordDecoder.Append("\n");

            // --- ГЕНЕРАЦИЯ ИНДЕКСОВ ЧТЕНИЯ ДЛЯ КАЖДОГО ТЕНЗОРА ---
            var indexCalculators = new StringBuilder();
            indexCalculators.Append("    // Вычисление смещений с учетом правил Broadcasting\n");
            foreach (var name in DiscoveredTensors)
            {
                var t = _tensorMeta[name];
                indexCalculators.Append($"    int idx_{name} = 0;\n");

                // Выравниваем оси тензора по правому краю относительно максимального ранга
                int rankDiff = maxRank - t.Shape.Length;

                for (int i = 0; i < t.Shape.Length; i++)
                {
                    int targetMaxDimIdx = i + rankDiff;

                    // ПРАВИЛО BROADCASTING: если размерность оси равна 1, 
                    // то индекс по этой оси всегда 0, шаг зануляется (умножение на 0)
                    if (t.Shape[i] == 1)
                    {
                        indexCalculators.Append($"    idx_{name} += 0 * {t.Strides[i]};\n");
                    }
                    else
                    {
                        // Иначе смещаемся на координату эталонной сетки, умноженную на stride оси тензора
                        indexCalculators.Append($"    idx_{name} += dim_{targetMaxDimIdx} * {t.Strides[i]};\n");
                    }
                }
                // Защита от выхода за физические границы памяти
                indexCalculators.Append($"    idx_{name} = min(idx_{name}, {t.Size} - 1);\n\n");
            }

            // --- СБОРКА СТРОКИ ЯДРА FORWARD ---
            var fSrc = new StringBuilder(utilities);
            fSrc.Append("kernel void fused_elementwise_kernel(\n");
            BuildSignature(fSrc);
            //fSrc.Append("    int tid = get_global_id(0);\n\n");
            fSrc.Append("int tid=get_global_size(1)*get_global_id(0)+get_global_id(1);\n\n");//XY оптимизация
            fSrc.Append(coordDecoder.ToString());
            fSrc.Append(indexCalculators.ToString());

            // Чтение в регистры по вычисленным индексам слияния размерностей
            fSrc.Append("    // Чтение во внутренние регистры\n");
            foreach (var name in DiscoveredTensors)
                fSrc.Append($"    float priv_{name} = {name}[idx_{name}];\n");

            fSrc.Append("\n    // Вычисления\n" + _bodyCode.ToString() + "\n");

            // Запись результатов во VRAM
            fSrc.Append("    // Сохранение результатов\n");
            foreach (var name in DiscoveredTensors)
            {
                fSrc.Append($"    if (tid < {name}_len) {{ {name}[idx_{name}] = priv_{name}; }}\n");
            }
            fSrc.Append("}");

            // --- СБОРКА СТРОКИ ЯДРА BACKWARD ---
            var bSrc = new StringBuilder(utilities);
            bSrc.Append("kernel void fused_elementwise_backward_kernel(\n");
            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    global float* {name}, global float* grad_{name}, global float* out_grad_{name}, const int {name}_len,\n");
            }
            bSrc.Length -= 2;
            bSrc.Append(") {\n");
            //bSrc.Append("    int tid = get_global_id(0);\n\n");
            bSrc.Append("int tid=get_global_size(1)*get_global_id(0)+get_global_id(1);\n\n");//XY оптимизация
            bSrc.Append(coordDecoder.ToString());
            bSrc.Append(indexCalculators.ToString());

            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    float priv_{name} = {name}[idx_{name}];\n");
                bSrc.Append($"    float priv_grad_{name} = grad_{name}[idx_{name}];\n");
                bSrc.Append($"    float priv_out_grad_{name} = 0.0f;\n");
            }
            bSrc.Append("\n    // Вычисление производных на регистрах\n");
            bSrc.Append(_backwardBodyCode.ToString());
            bSrc.Append("\n    // Сохранение рассчитанных градиентов\n");
            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    if (tid < {name}_len) {{ out_grad_{name}[idx_{name}] = priv_out_grad_{name}; }}\n");
            }
            bSrc.Append("}");

            return (fSrc.ToString(), bSrc.ToString());
        }

        private void BuildSignature(StringBuilder sb)
        {
            int idx = 0;
            foreach (var name in DiscoveredTensors)
            {
                sb.Append($"    global float* {name}, const int {name}_len");
                if (idx < DiscoveredTensors.Count - 1) sb.Append(",\n");
                idx++;
            }
            sb.Append(") {\n");
        }
       

        /// <summary>
        /// Список имен тензоров, в которые производилась запись (левая часть оператора '=').
        /// </summary>
        public HashSet<string> OutputTensorNames { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

   
        public override SyntaxNode VisitAssignmentExpression(AssignmentExpressionSyntax node)
        {
            string leftVar = node.Left.ToString().Trim();
            DiscoveredTensors.Add(leftVar);

            // ФИКСАЦИЯ ВЫХОДА: Маркируем тензор как выходной, так как в него идет запись
            OutputTensorNames.Add(leftVar);

            string opStr = node.OperatorToken.Text;
            _bodyCode.Append($"    priv_{leftVar} {opStr} ");
            ProcessRightSide(node.Right, leftVar);
            _bodyCode.Append(";\n");

            return node;
        }

        private void ProcessRightSide(ExpressionSyntax expression, string leftVar)
        {
            if (expression is InvocationExpressionSyntax invocation)
            {
                // 1. ПЕРЕХВАТ ФУНКЦИЙ (tanh, sin, cos, dropout)
                string funcText = invocation.Expression.ToString();
                var firstArgExpr = invocation.ArgumentList.Arguments[0].Expression;
                string argName = firstArgExpr.ToString().Trim();

                DiscoveredTensors.Add(argName);

                if (funcText.Contains("tanh"))
                {
                    _bodyCode.Append($"tanh(priv_{argName})");
                    _backwardBodyCode.Append($"    priv_out_grad_{argName} += priv_grad_{leftVar} * (1.0f - priv_{leftVar} * priv_{leftVar});\n");
                }
                else if (funcText.Contains("sin"))
                {
                    _bodyCode.Append($"sin(priv_{argName})");
                    _backwardBodyCode.Append($"    priv_out_grad_{argName} += priv_grad_{leftVar} * cos(priv_{argName});\n");
                }
                else if (funcText.Contains("cos"))
                {
                    _bodyCode.Append($"cos(priv_{argName})");
                    _backwardBodyCode.Append($"    priv_out_grad_{argName} += priv_grad_{leftVar} * (-sin(priv_{argName}));\n");
                }
                else if (funcText.Contains("dropout"))
                {
                    string pValStr = invocation.ArgumentList.Arguments[1].ToString();
                    string seedValStr = invocation.ArgumentList.Arguments[2].ToString();

                    _bodyCode.Append($"nn_dropout(priv_{argName}, {pValStr}f, tid, {seedValStr}u)");
                    _backwardBodyCode.Append($"    priv_out_grad_{argName} += nn_dropout_backward(priv_grad_{leftVar}, {pValStr}f, tid, {seedValStr}u);\n");
                }
            }
            else if (expression is BinaryExpressionSyntax binary)
            {
                // 2. ПЕРЕХВАТ БИНАРНОЙ МАТЕМАТИКИ (+, -, *, /)
                // Исправление 1: Рекурсивно раскрываем операнды, знаки подчеркивания добавятся внутри IdentifierNameSyntax автоматически!
                ProcessRightSide(binary.Left, leftVar);

                string op = binary.OperatorToken.Text;
                _bodyCode.Append($" {op} ");

                ProcessRightSide(binary.Right, leftVar);

                string leftStr = binary.Left.ToString().Trim();
                string rightStr = binary.Right.ToString().Trim();

                if (op == "/" && _externalConstants.TryGetValue(rightStr, out float val))
                {
                    string valStr = val.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "f";
                    _backwardBodyCode.Append($"    priv_out_grad_{leftStr} += priv_grad_{leftVar} / {valStr};\n");
                }
                else if (op == "+" && !_externalConstants.ContainsKey(leftStr) && !_externalConstants.ContainsKey(rightStr))
                {
                    // Исправление 2: Добавлены потерянные знаки подчеркивания '_' для обратного прохода сложения тензоров
                    if (DiscoveredTensors.Contains(leftStr))
                        _backwardBodyCode.Append($"    priv_out_grad_{leftStr} += priv_grad_{leftVar};\n");

                    if (DiscoveredTensors.Contains(rightStr))
                        _backwardBodyCode.Append($"    priv_out_grad_{rightStr} += priv_grad_{leftVar};\n");
                }
            }
            else if (expression is IdentifierNameSyntax identifier)
            {
                // 3. ПЕРЕХВАТ ИДЕНТИФИКАТОРОВ (Переменные-тензоры)
                string name = identifier.Identifier.Text.Trim();

                if (_externalConstants.TryGetValue(name, out float val))
                {
                    _bodyCode.Append(val.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "f");
                }
                else
                {
                    DiscoveredTensors.Add(name);
                    // Исправление 3: Строго добавляем префикс со знаком подчеркивания для прямого прохода
                    _bodyCode.Append($"priv_{name}");
                }
            }
            else
            {
                // 4. ПРЯМЫЕ ЧИСЛОВЫЕ КОНСТАНТЫ ИЗ ТЕКСТА
                string numStr = expression.ToString().Trim().Replace(",", ".");
                if (!numStr.Contains(".")) numStr += ".0";
                if (!numStr.EndsWith("f") && !numStr.EndsWith("u")) numStr += "f";
                _bodyCode.Append(numStr);
            }
        }

    }
}
