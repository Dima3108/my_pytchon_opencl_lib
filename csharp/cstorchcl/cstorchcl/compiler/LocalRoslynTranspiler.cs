using System;
using System.Collections.Generic;
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

        public (string forwardSource, string backwardSource) TranslateWithGrad(string pseudoCode, Dictionary<string, float> constants)
        {
            _externalConstants = constants ?? new Dictionary<string, float>();
            _bodyCode.Clear();
            _backwardBodyCode.Clear();
            DiscoveredTensors.Clear();

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

            // --- СБОРКА СТРОКИ ЯДРА FORWARD ---
            var fSrc = new StringBuilder(utilities);
            fSrc.Append("kernel void fused_elementwise_kernel(\n");
            BuildSignature(fSrc);
            fSrc.Append("    int tid = get_global_id(0);\n");
            BuildPrivateReads(fSrc);
            fSrc.Append("    // Вычисления\n" + _bodyCode.ToString() + "\n");
            BuildPrivateWrites(fSrc);

            // --- СБОРКА СТРОКИ ЯДРА BACKWARD ---
            // Чистим сигнатуру от конфликтующих модификаторов const
            var bSrc = new StringBuilder(utilities);
            bSrc.Append("kernel void fused_elementwise_backward_kernel(\n");
            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    global float* {name}, global float* grad_{name}, global float* out_grad_{name}, const int {name}_len,\n");
            }
            bSrc.Length -= 2; // Убираем запятую и перенос
            bSrc.Append(") {\n");
            bSrc.Append("    int tid = get_global_id(0);\n\n");

            // Инициализация локальных регистров для обратного прохода
            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    float priv_{name} = {name}[min(tid, {name}_len - 1)];\n");
                bSrc.Append($"    float priv_grad_{name} = grad_{name}[min(tid, {name}_len - 1)];\n");
                bSrc.Append($"    float priv_out_grad_{name} = 0.0f;\n");
            }
            bSrc.Append("\n    // Вычисление производных на регистрах\n");
            bSrc.Append(_backwardBodyCode.ToString());
            bSrc.Append("\n    // Сохранение рассчитанных градиентов во VRAM\n");
            foreach (var name in DiscoveredTensors)
            {
                bSrc.Append($"    if (tid < {name}_len) {{ out_grad_{name}[tid] = priv_out_grad_{name}; }}\n");
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

        private void BuildPrivateReads(StringBuilder sb)
        {
            foreach (var name in DiscoveredTensors)
                sb.Append($"    float priv_{name} = {name}[min(tid, {name}_len - 1)];\n");
            sb.Append("\n");
        }

        private void BuildPrivateWrites(StringBuilder sb)
        {
            foreach (var name in DiscoveredTensors)
                sb.Append($"    if (tid < {name}_len) {{ {name}[tid] = priv_{name}; }}\n");
            sb.Append("}");
        }

        public override SyntaxNode VisitAssignmentExpression(AssignmentExpressionSyntax node)
        {
            string leftVar = node.Left.ToString().Trim();
            DiscoveredTensors.Add(leftVar);
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
                    if (DiscoveredTensors.Contains(leftStr)) _backwardBodyCode.Append($"    priv_out_grad_{leftStr} += priv_grad_{leftVar};\n");
                    if (DiscoveredTensors.Contains(rightStr)) _backwardBodyCode.Append($"    priv_out_grad_{rightStr} += priv_grad_{leftVar};\n");
                }
            }
            else if (expression is IdentifierNameSyntax identifier)
            {
                string name = identifier.Identifier.Text.Trim();

                if (_externalConstants.TryGetValue(name, out float val))
                {
                    // Исправление: жестко форматируем константу как float с точкой (например 2.0f вместо 2f)
                    _bodyCode.Append(val.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "f");
                }
                else
                {
                    DiscoveredTensors.Add(name);
                    _bodyCode.Append($"priv_{name}");
                }
            }
            else
            {
                string numStr = expression.ToString().Trim().Replace(",", ".");
                if (!numStr.Contains(".")) numStr += ".0"; // Гарантируем точку для Си компилятора GPU
                if (!numStr.EndsWith("f") && !numStr.EndsWith("u")) numStr += "f";
                _bodyCode.Append(numStr);
            }
        }
    }
}

