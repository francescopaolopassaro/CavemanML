// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// The tiny transformer encoder shared by SynthelionML (keep/drop) and PrivacyGuardML (BIO tagging), implemented
// in plain C# so inference needs no torch and no native runtime:
//
//   word embedding + feature projection + positional embedding
//     -> N x TransformerEncoderLayer (post-norm, multi-head self-attention, GELU feed-forward)
//     -> Linear(n_out)
//
// Numerically it follows torch.nn.TransformerEncoderLayer(batch_first=True, activation="gelu", norm_first=False)
// in eval mode, so the logits agree with the reference PyTorch implementation to float32 rounding.
// ------------------------------------------------------------------------------
using System.Numerics;
using Caveman.ML.Weights;

namespace Caveman.ML.Model;

/// <summary>Hyper-parameters read from a model's <c>config.json</c>.</summary>
public sealed class EncoderConfig
{
    public int DModel { get; init; } = 128;
    public int Heads { get; init; } = 4;
    public int Layers { get; init; } = 2;
    public int FfnDim { get; init; } = 512;
    public int Features { get; init; } = 18;
    public int MaxSeqLen { get; init; } = 96;
    public int Outputs { get; init; } = 2;
}

public sealed class TinyEncoder
{
    private sealed class Linear
    {
        public readonly float[] W;   // [out, in] row-major
        public readonly float[] B;
        public readonly int In, Out;

        public Linear(Tensor w, Tensor b)
        {
            W = w.Data; B = b.Data; Out = w.Shape[0]; In = w.Shape[1];
        }

        /// <summary>y[t] = x[t]·Wᵀ + b for every row t.</summary>
        public float[] Apply(float[] x, int rows)
        {
            var y = new float[rows * Out];
            for (var t = 0; t < rows; t++)
            {
                var xi = x.AsSpan(t * In, In);
                for (var o = 0; o < Out; o++)
                {
                    var wi = W.AsSpan(o * In, In);
                    y[t * Out + o] = Dot(xi, wi) + B[o];
                }
            }
            return y;
        }
    }

    private sealed class Norm
    {
        public readonly float[] Gamma, Beta;
        public Norm(Tensor g, Tensor b) { Gamma = g.Data; Beta = b.Data; }

        /// <summary>LayerNorm over the last dimension (eps 1e-5, as torch's default).</summary>
        public void Apply(float[] x, int rows, int d)
        {
            for (var t = 0; t < rows; t++)
            {
                var row = x.AsSpan(t * d, d);
                double mean = 0;
                for (var i = 0; i < d; i++) mean += row[i];
                mean /= d;
                double var = 0;
                for (var i = 0; i < d; i++) { var diff = row[i] - mean; var += diff * diff; }
                var /= d;
                var inv = 1.0 / Math.Sqrt(var + 1e-5);
                for (var i = 0; i < d; i++) row[i] = (float)((row[i] - mean) * inv * Gamma[i] + Beta[i]);
            }
        }
    }

    private sealed class Layer
    {
        public Linear InProj = null!, OutProj = null!, Ff1 = null!, Ff2 = null!;
        public Norm Norm1 = null!, Norm2 = null!;
    }

    private readonly EncoderConfig _cfg;
    private readonly float[] _wordEmb;       // [vocab, d]
    private readonly int _wordRows;
    private readonly Linear _featProj;
    private readonly float[] _posEmb;        // [maxSeq, d]
    private readonly Layer[] _layers;
    private readonly Linear _head;

    public EncoderConfig Config => _cfg;
    public int Vocabulary => _wordRows;

    public TinyEncoder(EncoderConfig cfg, IReadOnlyDictionary<string, Tensor> w)
    {
        _cfg = cfg;
        var we = Get(w, "_word_emb.weight");
        _wordEmb = we.Data; _wordRows = we.Shape[0];
        _featProj = new Linear(Get(w, "_feat_proj.weight"), Get(w, "_feat_proj.bias"));
        _posEmb = Get(w, "_pos_emb.weight").Data;
        _layers = new Layer[cfg.Layers];
        for (var i = 0; i < cfg.Layers; i++)
        {
            var p = $"_encoder.layers.{i}.";
            _layers[i] = new Layer
            {
                InProj = new Linear(Get(w, p + "self_attn.in_proj_weight"), Get(w, p + "self_attn.in_proj_bias")),
                OutProj = new Linear(Get(w, p + "self_attn.out_proj.weight"), Get(w, p + "self_attn.out_proj.bias")),
                Ff1 = new Linear(Get(w, p + "linear1.weight"), Get(w, p + "linear1.bias")),
                Ff2 = new Linear(Get(w, p + "linear2.weight"), Get(w, p + "linear2.bias")),
                Norm1 = new Norm(Get(w, p + "norm1.weight"), Get(w, p + "norm1.bias")),
                Norm2 = new Norm(Get(w, p + "norm2.weight"), Get(w, p + "norm2.bias")),
            };
        }
        _head = new Linear(Get(w, "_head.weight"), Get(w, "_head.bias"));
    }

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out var t) ? t : throw new InvalidDataException($"model weights are missing '{key}'");

    /// <summary>
    /// Runs the encoder over one sequence (no padding needed: only real tokens are passed) and returns the
    /// output logits as <c>[length, Outputs]</c> row-major.
    /// </summary>
    /// <param name="wordIds">Word-vocabulary id, or char-n-gram bucket id + vocabulary size, for every token.</param>
    /// <param name="features">Per-token feature vectors, <c>[length, Features]</c> row-major.</param>
    public float[] Forward(int[] wordIds, float[] features)
    {
        var len = wordIds.Length;
        var d = _cfg.DModel;
        if (len > _cfg.MaxSeqLen) throw new ArgumentException($"sequence of {len} tokens exceeds max_seq_len {_cfg.MaxSeqLen}");

        var feat = _featProj.Apply(features, len);
        var x = new float[len * d];
        for (var t = 0; t < len; t++)
        {
            var id = wordIds[t];
            if (id < 0 || id >= _wordRows) id = 0;
            for (var i = 0; i < d; i++)
                x[t * d + i] = _wordEmb[id * d + i] + feat[t * d + i] + _posEmb[t * d + i];
        }

        foreach (var layer in _layers)
        {
            // post-norm: x = norm1(x + SelfAttention(x)); x = norm2(x + FeedForward(x))
            var attn = SelfAttention(layer, x, len);
            for (var i = 0; i < x.Length; i++) x[i] += attn[i];
            layer.Norm1.Apply(x, len, d);

            var h = layer.Ff1.Apply(x, len);
            for (var i = 0; i < h.Length; i++) h[i] = Gelu(h[i]);
            var f = layer.Ff2.Apply(h, len);
            for (var i = 0; i < x.Length; i++) x[i] += f[i];
            layer.Norm2.Apply(x, len, d);
        }

        return _head.Apply(x, len);
    }

    private float[] SelfAttention(Layer layer, float[] x, int len)
    {
        var d = _cfg.DModel;
        var heads = _cfg.Heads;
        var hd = d / heads;
        var qkv = layer.InProj.Apply(x, len);               // [len, 3d]: q | k | v
        var ctx = new float[len * d];
        var scores = new float[len];
        var scale = 1.0f / MathF.Sqrt(hd);

        for (var h = 0; h < heads; h++)
        {
            for (var i = 0; i < len; i++)
            {
                var q = qkv.AsSpan(i * 3 * d + h * hd, hd);
                var max = float.NegativeInfinity;
                for (var j = 0; j < len; j++)
                {
                    var k = qkv.AsSpan(j * 3 * d + d + h * hd, hd);
                    var s = Dot(q, k) * scale;
                    scores[j] = s;
                    if (s > max) max = s;
                }
                float sum = 0;
                for (var j = 0; j < len; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                var inv = 1f / sum;
                var o = ctx.AsSpan(i * d + h * hd, hd);
                for (var j = 0; j < len; j++)
                {
                    var wgt = scores[j] * inv;
                    var v = qkv.AsSpan(j * 3 * d + 2 * d + h * hd, hd);
                    for (var c = 0; c < hd; c++) o[c] += wgt * v[c];
                }
            }
        }
        return layer.OutProj.Apply(ctx, len);
    }

    private static float Gelu(float x) => 0.5f * x * (1f + (float)Erf(x / Math.Sqrt(2.0)));

    // Abramowitz-Stegun 7.1.26 is too coarse for parity with torch's exact GELU; use a high-precision erf.
    private static double Erf(double x)
    {
        // Numerical Recipes erfc Chebyshev fit, relative error < 1.2e-7 in float range - well below float32 rounding of the logits.
        var z = Math.Abs(x);
        var t = 1.0 / (1.0 + 0.5 * z);
        var r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - r : r - 1.0;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var n = a.Length;
        var i = 0;
        var acc = Vector<float>.Zero;
        var w = Vector<float>.Count;
        for (; i <= n - w; i += w) acc += new Vector<float>(a.Slice(i, w)) * new Vector<float>(b.Slice(i, w));
        var sum = Vector.Sum(acc);
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }
}
