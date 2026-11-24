using System;
using System.Collections.Generic;
using System.Linq;
using NOpenCL;
using cstorchcl.core;
using cstorchcl.compiler;

namespace cstorchcl.layers
{
    public class RwkvTimeMixing
    {
        private readonly NnClGlob _glob;
        private readonly string _id;

        public Tensor TimeMixK, TimeMixV, TimeMixR; // [C]
        public Tensor TimeDecayRaw;                 // [C] -> w = -exp(TimeDecayRaw)
        public Tensor TimeFirst;                    // [C] (u)

        public Linear KeyProj, ValueProj, ReceptanceProj, OutputProj;

        public RwkvTimeMixing(int dModel, ComputeContext ctx, NnClGlob glob, string layerId, Random rng)
        {
            _glob = glob;
            _id = $"{layerId}.tmix";

            TimeMixK = InitVector(dModel, ctx.Context, $"{_id}.mix_k", _ => 0.5f);
            TimeMixV = InitVector(dModel, ctx.Context, $"{_id}.mix_v", _ => 0.5f);
            TimeMixR = InitVector(dModel, ctx.Context, $"{_id}.mix_r", _ => 0.5f);

            // Инициализация затухания w и бонуса u по правилам RWKV
            TimeDecayRaw = InitVector(dModel, ctx.Context, $"{_id}.decay", c => -5.0f + 8.0f * (float)Math.Pow((double)c / dModel, 0.7));
            TimeFirst = InitVector(dModel, ctx.Context, $"{_id}.first", c => (float)(rng.NextDouble() * 0.2 - 0.1));

            KeyProj = new Linear(ctx,glob,dModel, dModel,  $"{_id}.k_proj");
            ValueProj = new Linear(ctx, glob, dModel, dModel,  $"{_id}.v_proj");
            ReceptanceProj = new Linear(ctx, glob, dModel, dModel, $"{_id}.r_proj");
            OutputProj = new Linear(ctx, glob, dModel, dModel,  $"{_id}.o_proj");
        }

        public Tensor Forward(Tensor x)
        {
            // 1. TokenShift (смешивание x_t и x_{t-1})
            Tensor xk = _glob.TokenShift(x, TimeMixK, $"{_id}.sh_k");
            Tensor xv = _glob.TokenShift(x, TimeMixV, $"{_id}.sh_v");
            Tensor xr = _glob.TokenShift(x, TimeMixR, $"{_id}.sh_r");

            // 2. Линейные проекции K, V, R
            Tensor k = KeyProj.Forward(xk);
            Tensor v = ValueProj.Forward(xv);
            Tensor r = ReceptanceProj.Forward(xr);

            // 3. Активация гейта receptance = sigmoid(r) и отрицательное затухание w = -exp(w_raw)
            Tensor sr = _glob.RwkvUnary(r, mode: 0, $"{_id}.sig_r");
            Tensor w = _glob.RwkvUnary(TimeDecayRaw, mode: 2, $"{_id}.negexp_w");

            // 4. Рекуррентное внимание WKV
            Tensor wkv = _glob.Wkv(w, TimeFirst, k, v, $"{_id}.wkv");

            // 5. Гейтирование и выходная проекция: OutputProj(sigmoid(r) * wkv)
            Tensor rwkv = _glob.Mul(sr, wkv, $"{_id}.gate");
            return OutputProj.Forward(rwkv);
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return TimeMixK;
            yield return TimeMixV;
            yield return TimeMixR;
            yield return TimeDecayRaw;
            yield return TimeFirst;
            foreach (var p in KeyProj.GetParameters()) yield return p;
            foreach (var p in ValueProj.GetParameters()) yield return p;
            foreach (var p in ReceptanceProj.GetParameters()) yield return p;
            foreach (var p in OutputProj.GetParameters()) yield return p;
        }

        private static Tensor InitVector(int size, Context ctx, string id, Func<int, float> initFn)
        {
            float[] data = new float[size];
            for (int i = 0; i < size; i++) data[i] = initFn(i);
            return new Tensor(data, new[] { size }, ctx, id, requiresGrad: true);
        }
    }

    public class RwkvChannelMixing
    {
        private readonly NnClGlob _glob;
        private readonly string _id;

        public Tensor TimeMixK, TimeMixR; // [C]
        public Linear KeyProj, ValueProj, ReceptanceProj;

        public RwkvChannelMixing(int dModel, int dFfn, ComputeContext ctx, NnClGlob glob, string layerId)
        {
            _glob = glob;
            _id = $"{layerId}.cmix";

            float[] half = Enumerable.Repeat(0.5f, dModel).ToArray();
            TimeMixK = new Tensor(half, new[] { dModel }, ctx.Context, $"{_id}.mix_k", requiresGrad: true);
            TimeMixR = new Tensor(half, new[] { dModel }, ctx.Context, $"{_id}.mix_r", requiresGrad: true);

            KeyProj = new Linear(ctx, glob, dModel, dFfn,  $"{_id}.k_proj");
            ValueProj = new Linear(ctx, glob, dFfn, dModel,  $"{_id}.v_proj");
            ReceptanceProj = new Linear(ctx, glob, dModel, dModel,  $"{_id}.r_proj");
        }

        public Tensor Forward(Tensor x)
        {
            Tensor xk = _glob.TokenShift(x, TimeMixK, $"{_id}.sh_k");
            Tensor xr = _glob.TokenShift(x, TimeMixR, $"{_id}.sh_r");

            // k = relu(KeyProj(xk))^2
            Tensor kPre = KeyProj.Forward(xk);
            Tensor kAct = _glob.RwkvUnary(kPre, mode: 1, $"{_id}.sqrelu"); // Squared ReLU
            Tensor kv = ValueProj.Forward(kAct);

            // r = sigmoid(ReceptanceProj(xr))
            Tensor rPre = ReceptanceProj.Forward(xr);
            Tensor rGate = _glob.RwkvUnary(rPre, mode: 0, $"{_id}.sig_r");

            return _glob.Mul(rGate, kv, $"{_id}.out");
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return TimeMixK;
            yield return TimeMixR;
            foreach (var p in KeyProj.GetParameters()) yield return p;
            foreach (var p in ValueProj.GetParameters()) yield return p;
            foreach (var p in ReceptanceProj.GetParameters()) yield return p;
        }
    }

    public class RwkvBlock
    {
        private readonly NnClGlob _glob;
        private readonly string _id;

        public Tensor Gamma1, Gamma2; // Веса для RMSNorm
        public RwkvTimeMixing TimeMix;
        public RwkvChannelMixing ChannelMix;

        public RwkvBlock(int dModel, int dFfn, ComputeContext ctx, NnClGlob glob, string layerId, Random rng)
        {
            _glob = glob;
            _id = layerId;

            float[] ones = Enumerable.Repeat(1.0f, dModel).ToArray();
            Gamma1 = new Tensor(ones, new[] { dModel }, ctx.Context, $"{_id}.ln1_gamma", requiresGrad: true);
            Gamma2 = new Tensor(ones, new[] { dModel }, ctx.Context, $"{_id}.ln2_gamma", requiresGrad: true);

            TimeMix = new RwkvTimeMixing(dModel, ctx, glob, _id, rng);
            ChannelMix = new RwkvChannelMixing(dModel, dFfn, ctx, glob, _id);
        }

        public Tensor Forward(Tensor x)
        {
            // Блок 1: x = x + TimeMix(RMSNorm(x))
            Tensor norm1 = _glob.RmsNorm(x, Gamma1, $"{_id}.ln1");
            Tensor tmOut = TimeMix.Forward(norm1);
            Tensor x1 = _glob.Add(x, tmOut); // или ваш AddOp

            // Блок 2: x = x + ChannelMix(RMSNorm(x))
            Tensor norm2 = _glob.RmsNorm(x1, Gamma2, $"{_id}.ln2");
            Tensor cmOut = ChannelMix.Forward(norm2);
            Tensor x2 = _glob.Add(x1, cmOut);

            return x2;
        }

        public List<Tensor> Parameters()
        {
            var list = new List<Tensor> { Gamma1, Gamma2 };
            list.AddRange(TimeMix.Parameters());
            list.AddRange(ChannelMix.Parameters());
            return list;
        }
    }
}
