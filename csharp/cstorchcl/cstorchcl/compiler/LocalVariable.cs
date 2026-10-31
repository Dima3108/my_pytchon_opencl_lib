using System;
using cstorchcl.core;

namespace cstorchcl.compiler
{
    public class LocalVariable
    {
        public Tensor Tensor { get; }
        public string Name { get; }

        public LocalVariable(Tensor tensor, string name)
        {
            Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
            Name = name;
        }

        /// <summary>
        /// Эквивалент оператора присваивания (=) для обхода ограничений Expression Trees.
        /// </summary>
        public LocalVariable Assign(LocalVariable value) => this;
        public LocalVariable Assign(double value) => this;

        // Перегруженные операторы теперь просто участвуют в построении математического дерева
        public static LocalVariable operator +(LocalVariable a, double b) => a;
        public static LocalVariable operator -(LocalVariable a, double b) => a;
        public static LocalVariable operator *(LocalVariable a, double b) => a;
        public static LocalVariable operator /(LocalVariable a, double b) => a;

        public static LocalVariable operator +(LocalVariable a, LocalVariable b) => a;
        public static LocalVariable operator -(LocalVariable a, LocalVariable b) => a;
        public static LocalVariable operator *(LocalVariable a, LocalVariable b) => a;
        public static LocalVariable operator /(LocalVariable a, LocalVariable b) => a;
    }

    public static class nnclloc
    {
        public static LocalVariable sin(LocalVariable v) => v;
        public static LocalVariable cos(LocalVariable v) => v;
        public static LocalVariable tanh(LocalVariable v) => v;
        public static LocalVariable dropout(LocalVariable v, float p, uint seed) => v;
    }
}
