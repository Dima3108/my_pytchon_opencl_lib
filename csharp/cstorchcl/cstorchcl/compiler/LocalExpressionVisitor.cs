using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Text;

namespace cstorchcl.compiler
{
    public class LocalExpressionVisitor : ExpressionVisitor
    {
        private readonly StringBuilder _bodyCode = new StringBuilder();

        // Отслеживаем только используемые тензоры для сигнатуры аргументов ядра
        public Dictionary<string, LocalVariable> UsedTensors { get; } = new Dictionary<string, LocalVariable>();

        /// <summary>
        /// Генерирует оптимизированный исходный код OpenCL ядра с подставленными константами и функциями дропаута.
        /// </summary>
        public string GenerateKernelSource()
        {
            var source = new StringBuilder();

            // Встраиваем ваш быстрый генератор случайных чисел и логику дропаута в начало кода
            source.Append(@"// Встроенные утилиты генерации шума и Dropout
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
    if (r < p) {
        return 0.0f;
    } else {
        return x / (1.0f - p);
    }
}

");

            // Формируем сигнатуру ядра. Теперь здесь ТОЛЬКО буферы и их длины!
            source.Append("kernel void fused_elementwise_kernel(\n");

            int tensorIdx = 0;
            foreach (var kvp in UsedTensors)
            {
                source.Append($"    global float* {kvp.Key}, const int {kvp.Key}_len");
                if (tensorIdx < UsedTensors.Count - 1) source.Append(",\n");
                tensorIdx++;
            }

            source.Append(") {\n");
            source.Append("    int tid = get_global_id(0);\n\n");

            // Оптимизация private-памяти (регистров GPU)
            source.Append("    // Чтение во внутренние регистры\n");
            foreach (var kvp in UsedTensors)
            {
                source.Append($"    float priv_{kvp.Key} = {kvp.Key}[min(tid, {kvp.Key}_len - 1)];\n");
            }
            source.Append("\n");

            // Тело вычислений
            source.Append("    // Вычисления над регистрами\n");
            source.Append(_bodyCode.ToString());
            source.Append("\n");

            // Безопасное сохранение результатов во VRAM
            source.Append("    // Сохранение результатов во VRAM\n");
            foreach (var kvp in UsedTensors)
            {
                source.Append($"    if (tid < {kvp.Key}_len) {{ {kvp.Key}[tid] = priv_{kvp.Key}; }}\n");
            }
            source.Append("}");

            return source.ToString();
        }



        protected override Expression VisitMethodCall(MethodCallExpression node)
        {
            // 1. Обработка метода .Assign(...) -> замена знака равенства '='
            if (node.Method.Name == "Assign" && node.Object is MemberExpression leftVariable)
            {
                string varName = leftVariable.Member.Name;
                _bodyCode.Append($"    priv_{varName} = ");

                // Разбираем то, что передано внутрь Assign(...)
                Visit(node.Arguments[0]);
                _bodyCode.Append(";\n");
                return node;
            }

            // 2. Обработка стандартных математических функций активации
            if (node.Method.DeclaringType == typeof(nnclloc))
            {
                string funcName = node.Method.Name.ToLower();

                if (funcName == "dropout")
                {
                    _bodyCode.Append("nn_dropout(");
                    if (node.Arguments[0] is MemberExpression varMember)
                    {
                        _bodyCode.Append($"priv_{varMember.Member.Name}, ");
                    }
                    var pLambda = Expression.Lambda<Func<float>>(node.Arguments[1]);
                    float pVal = pLambda.Compile()();
                    _bodyCode.Append(pVal.ToString(System.Globalization.CultureInfo.InvariantCulture) + "f, tid, ");

                    var seedLambda = Expression.Lambda<Func<uint>>(node.Arguments[2]);
                    uint seedVal = seedLambda.Compile()();
                    _bodyCode.Append(seedVal + "u)");
                }
                else
                {
                    _bodyCode.Append($"{funcName}(");
                    if (node.Arguments[0] is MemberExpression argMember)
                    {
                        _bodyCode.Append($"priv_{argMember.Member.Name}");
                    }
                    _bodyCode.Append(")");
                }
                return node;
            }

            return base.VisitMethodCall(node);
        }

        // Переписываем VisitBinary для обработки простых операций внутри Assign (например, A / q)
        protected override Expression VisitBinary(BinaryExpression node)
        {
            Visit(node.Left);

            string opStr = node.NodeType switch
            {
                ExpressionType.Add => " + ",
                ExpressionType.Subtract => " - ",
                ExpressionType.Multiply => " * ",
                ExpressionType.Divide => " / ",
                _ => throw new NotSupportedException($"Операция {node.NodeType} не поддерживается в JIT.")
            };

            _bodyCode.Append(opStr);
            Visit(node.Right);

            return node;
        }

        // Обновляем VisitMember, чтобы он правильно подставлял имена регистров в правых частях уравнений
        protected override Expression VisitMember(MemberExpression node)
        {
            if (node.Type == typeof(LocalVariable))
            {
                string name = node.Member.Name;
                if (!UsedTensors.ContainsKey(name))
                {
                    var objectMember = Expression.Convert(node, typeof(object));
                    var getterLambda = Expression.Lambda<Func<object>>(objectMember);
                    var localBuffer = (LocalVariable)getterLambda.Compile()();
                    UsedTensors.Add(name, localBuffer);
                }

                // Если мы уже внутри парсинга выражения, то подставляем имя приватного регистра GPU
                _bodyCode.Append($"priv_{name}");
                return node;
            }

            // Если это внешняя переменная типа float q
            if (node.Type == typeof(float) || node.Type == typeof(double))
            {
                var objectMember = Expression.Convert(node, typeof(object));
                var getterLambda = Expression.Lambda<Func<object>>(objectMember);
                float val = Convert.ToSingle(getterLambda.Compile()());

                _bodyCode.Append(val.ToString(System.Globalization.CultureInfo.InvariantCulture) + "f");
                return node;
            }

            return base.VisitMember(node);
        }

    }
}
