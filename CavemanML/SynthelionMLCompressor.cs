// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// SynthelionML - learned per-token keep/drop prompt compressor. A small, fully offline transformer encoder
// trained on Wikipedia corpora in 39 languages to predict which words can be dropped while preserving meaning
// (self-distillation from the rule-based Aggressive compressor, so it generalises beyond the rules by attending
// to the surrounding words). Port of Synthelion 1.2.5's synthelionml.py with a pure-C# inference engine.
//
// Inference: tokenize -> keep numbers, URLs, e-mails, proper nouns, negations and structured values verbatim ->
// score every other word with the model -> rank-based ratio controller drops the lowest-scoring words until the
// target compression is reached (70% by default) -> never leave a sentence empty -> rebuild the text.
// ------------------------------------------------------------------------------
using System.Text.Json;
using caveman.core;
using caveman.core.entities;
using Caveman.ML.Model;
using Caveman.ML.Text;
using Caveman.ML.Weights;

namespace Caveman.ML;

/// <summary>Learned prompt compressor (SynthelionML). Thread-safe; load once and reuse.</summary>
public sealed class SynthelionMLCompressor
{
    private static readonly FunctionWordProvider Provider = new();
    private static readonly CavemanLanguageDetector Detector = new();

    private readonly TinyEncoder _encoder;
    private readonly MlVocab _vocab;
    private readonly object _lock = new();

    /// <summary>Probability above which a word is kept when no ratio or budget is requested.</summary>
    public double KeepThreshold { get; }

    /// <summary>The compression ratio the rank controller aims for by default (0.7 = drop at least 70% of the words).</summary>
    public double MinCompression { get; }

    public string ModelDirectory { get; }

    private SynthelionMLCompressor(string dir, TinyEncoder encoder, MlVocab vocab, double keepThreshold, double minCompression)
    {
        ModelDirectory = dir;
        _encoder = encoder;
        _vocab = vocab;
        KeepThreshold = keepThreshold;
        MinCompression = minCompression;
    }

    // ───────────────────────────── loading ─────────────────────────────

    /// <summary>Loads the model from a local directory (or the default locations). Returns <c>null</c> when it is not installed.</summary>
    public static SynthelionMLCompressor? Load(string? directory = null)
    {
        var dir = ModelStore.Resolve(MlModelKind.SynthelionML, directory);
        return dir is null ? null : FromDirectory(dir);
    }

    /// <summary>Loads the model, downloading it from Hugging Face on first use when it is not installed.</summary>
    public static async Task<SynthelionMLCompressor?> LoadAsync(string? directory = null, bool download = true,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var dir = download
            ? await ModelStore.EnsureAsync(MlModelKind.SynthelionML, directory, progress, ct).ConfigureAwait(false)
            : ModelStore.Resolve(MlModelKind.SynthelionML, directory);
        return dir is null ? null : FromDirectory(dir);
    }

    public static SynthelionMLCompressor FromDirectory(string dir)
    {
        using var cfgDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "config.json")));
        var c = cfgDoc.RootElement;
        int Int(string k, int d) => c.TryGetProperty(k, out var v) ? v.GetInt32() : d;
        double Dbl(string k, double d) => c.TryGetProperty(k, out var v) ? v.GetDouble() : d;

        var cfg = new EncoderConfig
        {
            DModel = Int("d_model", 128), Heads = Int("n_heads", 4), Layers = Int("n_layers", 2),
            FfnDim = Int("ffn_dim", 512), Features = Int("n_features", 18), MaxSeqLen = Int("max_seq_len", 512), Outputs = 2,
        };

        var weights = Directory.EnumerateFiles(dir, "model.safetensors").FirstOrDefault()
                      ?? Path.Combine(dir, "model.bin");
        var encoder = new TinyEncoder(cfg, TensorFiles.Load(weights));
        var vocab = MlVocab.Load(Path.Combine(dir, "vocab.json"));
        return new SynthelionMLCompressor(dir, encoder, vocab, Dbl("keep_threshold", 0.5), Dbl("min_compression", 0.0));
    }

    // ───────────────────────────── scoring ─────────────────────────────

    private static float[] WordFeatures(string word, bool isStopword)
    {
        // [is_stopword, is_capitalized, length_bucket_0..15]
        var f = new float[18];
        f[0] = isStopword ? 1f : 0f;
        f[1] = word.Length > 0 && char.IsUpper(word[0]) ? 1f : 0f;
        f[2 + Math.Min(word.Length, 15)] = 1f;
        return f;
    }

    /// <summary>Keep probability (0..1) of every word, in order.</summary>
    public IReadOnlyList<double> ScoreWords(IReadOnlyList<string> words, string iso3)
    {
        var fw = Provider.GetFunctionWords(iso3);
        return ScoreWords(words, w => fw.Contains(w.ToLowerInvariant()));
    }

    /// <summary>Keep probability of every word, with an explicit stop-word predicate (the model's first input feature).</summary>
    public IReadOnlyList<double> ScoreWords(IReadOnlyList<string> words, Func<string, bool> isStopword)
    {
        var maxSeq = _encoder.Config.MaxSeqLen;
        var scores = new List<double>(words.Count);
        for (var start = 0; start < words.Count; start += maxSeq)
        {
            var n = Math.Min(maxSeq, words.Count - start);
            var ids = new int[n];
            var feats = new float[n * _encoder.Config.Features];
            for (var i = 0; i < n; i++)
            {
                var w = words[start + i];
                ids[i] = _vocab.Encode(w);
                WordFeatures(w, isStopword(w)).CopyTo(feats, i * _encoder.Config.Features);
            }
            float[] logits;
            lock (_lock) logits = _encoder.Forward(ids, feats);
            for (var i = 0; i < n; i++)
            {
                // softmax over {drop, keep}; probability of "keep" (index 1)
                double drop = logits[i * 2], keep = logits[i * 2 + 1];
                scores.Add(1.0 / (1.0 + Math.Exp(drop - keep)));
            }
        }
        return scores;
    }

    // ───────────────────────────── compression ─────────────────────────────

    private static readonly Dictionary<string, HashSet<string>> Negation = new(StringComparer.OrdinalIgnoreCase)
    {
        ["eng"] = new(StringComparer.OrdinalIgnoreCase) { "not", "n't", "no", "never", "none", "nobody", "nothing", "neither", "nor" },
        ["ita"] = new(StringComparer.OrdinalIgnoreCase) { "non", "no", "né", "neanche", "nemmeno", "manco" },
        ["fra"] = new(StringComparer.OrdinalIgnoreCase) { "ne", "pas", "non", "jamais", "aucun", "aucune", "rien", "personne" },
        ["spa"] = new(StringComparer.OrdinalIgnoreCase) { "no", "nunca", "jamás", "ni", "nadie", "nada", "ningún", "ninguna", "ninguno" },
        ["deu"] = new(StringComparer.OrdinalIgnoreCase) { "nicht", "kein", "keine", "keiner", "keinem", "keinen", "keines", "nie", "niemals", "nichts", "niemand" },
        ["por"] = new(StringComparer.OrdinalIgnoreCase) { "não", "nao", "nunca", "jamais", "nem", "nenhum", "nenhuma", "ninguém" },
        ["zho"] = new(StringComparer.OrdinalIgnoreCase) { "不", "没", "没有", "别", "无", "非", "未", "莫" },
    };

    private static bool IsNegation(string word, string iso3)
    {
        if (!Negation.TryGetValue(iso3, out var neg)) return false;
        var lower = word.ToLowerInvariant();
        return neg.Contains(lower) || (iso3 == "zho" && neg.Any(p => lower.StartsWith(p, StringComparison.OrdinalIgnoreCase)));
    }

    private static bool IsNumber(string text)
    {
        var stripped = text.Replace(",", "").Replace(".", "").Replace("-", "").Replace("+", "");
        return stripped.Length > 0 && stripped.All(char.IsDigit);
    }

    private static bool LooksLikeUrlOrEmail(string text)
    {
        var t = text.ToLowerInvariant();
        return t.Contains('@') || t.StartsWith("http", StringComparison.Ordinal) || t.StartsWith("www.", StringComparison.Ordinal)
               || t.EndsWith(".com", StringComparison.Ordinal) || t.EndsWith(".it", StringComparison.Ordinal)
               || t.EndsWith(".org", StringComparison.Ordinal) || t.EndsWith(".net", StringComparison.Ordinal)
               || t.EndsWith(".io", StringComparison.Ordinal);
    }

    private static bool IsProper(string word, HashSet<string>? proper)
    {
        if (word.Length > 0 && char.IsUpper(word[0]) && !word.All(c => !char.IsLetter(c) || char.IsUpper(c))) return true;
        return proper is { Count: > 0 } && proper.Contains(word.ToLowerInvariant());
    }

    private static HashSet<string> ProperNouns(string iso3)
    {
        var data = Provider.LoadWordData(iso3);
        return data is null ? new HashSet<string>() : new HashSet<string>(data.proper_nouns.Select(p => p.ToLowerInvariant()));
    }

    /// <summary>
    /// Compresses <paramref name="text"/>. <paramref name="budget"/> caps the number of surviving words;
    /// <paramref name="minCompression"/> (0..1) guarantees a minimum compression ratio by dropping the lowest-scoring
    /// words first. With neither, the model's own <see cref="MinCompression"/> target applies.
    /// </summary>
    public string Compress(string text, string? iso3 = null, int? budget = null, double? minCompression = null)
    {
        if (string.IsNullOrWhiteSpace(text)) return text ?? "";
        var lang = iso3 ?? Detector.Detect(text);
        var tokens = MlTokenizer.Tokenize(text);
        var mc = minCompression ?? (MinCompression > 0 ? MinCompression : (double?)null);
        return MlTokenizer.Join(CompressTokens(tokens, lang, budget, mc));
    }

    /// <summary>Compresses and reports token accounting in the same shape as the rule-based levels.</summary>
    public CompressionResult CompressResult(string text, string? iso3 = null, int? budget = null, double? minCompression = null)
    {
        var lang = iso3 ?? Detector.Detect(text);
        var tokens = MlTokenizer.Tokenize(text);
        var mc = minCompression ?? (MinCompression > 0 ? MinCompression : (double?)null);
        var kept = CompressTokens(tokens, lang, budget, mc);
        return new CompressionResult
        {
            CompressedText = MlTokenizer.Join(kept),
            OriginalTokens = tokens.Count(t => !t.IsPunct),
            CompressedTokens = kept.Count,
        };
    }

    private List<string> CompressTokens(List<MlToken> tokens, string iso3, int? budget, double? minCompression)
    {
        var keep = new HashSet<int>();
        var wordIdx = new List<int>();
        var words = new List<string>();
        var nWords = 0;
        var proper = ProperNouns(iso3);

        for (var i = 0; i < tokens.Count; i++)
        {
            var tok = tokens[i];
            if (tok.IsPunct) continue;
            nWords++;
            if (tok.Protected || IsNumber(tok.Text) || LooksLikeUrlOrEmail(tok.Text) || IsProper(tok.Text, proper) || IsNegation(tok.Text, iso3))
            {
                keep.Add(i);
                continue;
            }
            wordIdx.Add(i);
            words.Add(tok.Text);
        }

        if (words.Count > 0)
        {
            var scores = ScoreWords(words, iso3);

            if (budget is > 0)
            {
                // priority 1: an explicit cap on the number of surviving words
                var room = Math.Max(0, budget.Value - keep.Count);
                double threshold;
                if (room > 0)
                {
                    var order = Enumerable.Range(0, scores.Count).OrderByDescending(j => scores[j]).ToList();
                    threshold = scores[order[Math.Min(room, order.Count) - 1]];
                }
                else threshold = double.PositiveInfinity;
                for (var j = 0; j < wordIdx.Count; j++) if (scores[j] >= threshold) keep.Add(wordIdx[j]);
            }
            else if (minCompression is > 0)
            {
                // priority 2: rank-based ratio controller - keep the highest-scoring words until the target ratio is met
                var targetKeep = Math.Max(1, (int)Math.Ceiling((1.0 - minCompression.Value) * nWords));
                var room = Math.Max(0, targetKeep - keep.Count);
                if (room > 0)
                {
                    var order = Enumerable.Range(0, scores.Count).OrderByDescending(j => scores[j]).ToList();
                    foreach (var j in order.Take(room)) keep.Add(wordIdx[j]);
                }
            }
            else
            {
                for (var j = 0; j < wordIdx.Count; j++) if (scores[j] >= KeepThreshold) keep.Add(wordIdx[j]);
            }
        }

        ApplySentenceFloor(tokens, keep);
        return tokens.Where((_, i) => keep.Contains(i)).Select(t => t.Text).ToList();
    }

    private static readonly HashSet<string> SentenceEnders = new() { ".", "!", "?", ";", ":", "\n" };

    /// <summary>Guarantees at least one word survives in every sentence.</summary>
    private static void ApplySentenceFloor(List<MlToken> tokens, HashSet<int> keep)
    {
        var start = 0;
        for (var i = 1; i < tokens.Count; i++)
        {
            if (tokens[i].IsPunct && SentenceEnders.Contains(tokens[i].Text))
            {
                EnsureOne(tokens, keep, start, i);
                start = i + 1;
            }
        }
        EnsureOne(tokens, keep, start, tokens.Count);
    }

    private static void EnsureOne(List<MlToken> tokens, HashSet<int> keep, int start, int end)
    {
        for (var i = start; i < end; i++) if (keep.Contains(i)) return;
        for (var i = start; i < end; i++)
            if (!tokens[i].IsPunct) { keep.Add(i); return; }
    }
}
