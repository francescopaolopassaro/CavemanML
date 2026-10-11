// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// PrivacyGuardML - Synthelion's own PII confirmation model. A tiny offline transformer encoder with a per-token
// BIO tagging head over every PrivacyGuard category (phone, e-mail, card, IBAN, national/tax ID, SSN, GPS, plate,
// badge, business ID, secret, handle, legal case, booking code, minor age), trained on synthetic values generated
// from the checksum validators and rule patterns of Caveman.PrivacyGuard, inserted into real multilingual sentences.
//
// It is NOT a replacement for PrivacyAnalyzer's regex + checksum core, which stays the default. Plugged into
// PrivacyAnalyzer.MlDetector it is an opt-in confirmation signal: a value that matches a requires_context rule but
// has no context keyword nearby is confirmed by an overlapping prediction of the right label. A failed checksum
// still vetoes detection whatever the model says - it can only recover recall, never add a false positive.
// ------------------------------------------------------------------------------
using System.Text.Json;
using Caveman.ML.Model;
using Caveman.ML.Text;
using Caveman.ML.Weights;
using Caveman.PrivacyGuard;

namespace Caveman.ML;

/// <summary>Learned PII detector (PrivacyGuardML), usable as <see cref="PrivacyAnalyzer.MlDetector"/>.</summary>
public sealed class PrivacyGuardMLDetector : IMlEntityDetector
{
    private static readonly Dictionary<string, string> LabelOf = new(StringComparer.Ordinal)
    {
        ["PHONE"] = "phone number", ["EMAIL"] = "email address", ["CREDITCARD"] = "credit card number", ["IBAN"] = "iban number",
        ["NATIONALID"] = "national identification number", ["TAXID"] = "tax identification number", ["SSN"] = "social security number",
        ["GPS"] = "gps coordinates", ["VEHICLEPLATE"] = "vehicle license plate", ["BADGEID"] = "employee badge id",
        ["BUSINESSID"] = "business identification number", ["SECRET"] = "credential or secret", ["SOCIALHANDLE"] = "social media handle",
        ["LEGALCASE"] = "legal case number", ["PNRCODE"] = "booking reference", ["MINORDATA"] = "minor age indicator",
    };

    private readonly TinyEncoder _encoder;
    private readonly MlVocab _vocab;
    private readonly string[] _tags;
    private readonly object _lock = new();

    /// <summary>Spans below this mean confidence are dropped (default 0.6).</summary>
    public double MinConfidence { get; set; }

    public string ModelDirectory { get; }

    private PrivacyGuardMLDetector(string dir, TinyEncoder encoder, MlVocab vocab, string[] tags, double minConfidence)
    {
        ModelDirectory = dir; _encoder = encoder; _vocab = vocab; _tags = tags; MinConfidence = minConfidence;
    }

    // ───────────────────────────── loading ─────────────────────────────

    public static PrivacyGuardMLDetector? Load(string? directory = null, double minConfidence = 0.6)
    {
        var dir = ModelStore.Resolve(MlModelKind.PrivacyGuardML, directory);
        return dir is null ? null : FromDirectory(dir, minConfidence);
    }

    public static async Task<PrivacyGuardMLDetector?> LoadAsync(string? directory = null, bool download = true, double minConfidence = 0.6,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var dir = download
            ? await ModelStore.EnsureAsync(MlModelKind.PrivacyGuardML, directory, progress, ct).ConfigureAwait(false)
            : ModelStore.Resolve(MlModelKind.PrivacyGuardML, directory);
        return dir is null ? null : FromDirectory(dir, minConfidence);
    }

    public static PrivacyGuardMLDetector FromDirectory(string dir, double minConfidence = 0.6)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "config.json")));
        var c = doc.RootElement;
        int Int(string k, int d) => c.TryGetProperty(k, out var v) ? v.GetInt32() : d;
        var tags = c.GetProperty("tags").EnumerateArray().Select(t => t.GetString()!).ToArray();

        var cfg = new EncoderConfig
        {
            DModel = Int("d_model", 128), Heads = Int("n_heads", 4), Layers = Int("n_layers", 2), FfnDim = Int("ffn_dim", 512),
            Features = Int("n_features", 21), MaxSeqLen = Int("max_seq_len", 96), Outputs = tags.Length,
        };
        var weights = Directory.EnumerateFiles(dir, "model.safetensors").FirstOrDefault() ?? Path.Combine(dir, "model.bin");
        return new PrivacyGuardMLDetector(dir, new TinyEncoder(cfg, TensorFiles.Load(weights)), MlVocab.Load(Path.Combine(dir, "vocab.json")), tags, minConfidence);
    }

    // ───────────────────────────── inference ─────────────────────────────

    /// <summary>Shape-oriented features: [has_digit, has_at, has_plus, has_separator, is_upper, length_bucket_0..15].</summary>
    private static float[] WordFeatures(string word)
    {
        var f = new float[21];
        f[0] = word.Any(char.IsDigit) ? 1f : 0f;
        f[1] = word.Contains('@') ? 1f : 0f;
        f[2] = word.StartsWith('+') ? 1f : 0f;
        f[3] = word.IndexOfAny(new[] { '-', '.', '/' }) >= 0 ? 1f : 0f;
        var hasCased = word.Any(char.IsLetter);
        f[4] = hasCased && word.Length > 1 && !word.Any(char.IsLower) ? 1f : 0f;
        f[5 + Math.Min(word.Length, 15)] = 1f;
        return f;
    }

    /// <summary>Per-word tag and confidence, for diagnostics and parity tests.</summary>
    public (string[] Tags, double[] Confidences) Tag(IReadOnlyList<string> words)
    {
        var n = Math.Min(words.Count, _encoder.Config.MaxSeqLen);
        var nf = _encoder.Config.Features;
        var ids = new int[n];
        var feats = new float[n * nf];
        for (var i = 0; i < n; i++)
        {
            ids[i] = _vocab.Encode(words[i]);
            WordFeatures(words[i]).CopyTo(feats, i * nf);
        }
        float[] logits;
        lock (_lock) logits = _encoder.Forward(ids, feats);

        var tags = new string[n];
        var conf = new double[n];
        var k = _tags.Length;
        for (var i = 0; i < n; i++)
        {
            double max = double.NegativeInfinity;
            var arg = 0;
            for (var j = 0; j < k; j++) if (logits[i * k + j] > max) { max = logits[i * k + j]; arg = j; }
            double sum = 0;
            for (var j = 0; j < k; j++) sum += Math.Exp(logits[i * k + j] - max);
            tags[i] = _tags[arg];
            conf[i] = 1.0 / sum;     // softmax probability of the arg-max tag
        }
        return (tags, conf);
    }

    /// <summary>Finds PII entities in <paramref name="text"/>. Never throws: any failure yields an empty list (regex-only analysis).</summary>
    public IReadOnlyList<MlSpan> Detect(string text)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(text)) return Array.Empty<MlSpan>();
            var words = MlTokenizer.Tokenize(text).Where(t => !t.IsPunct).Select(t => t.Text).ToList();
            if (words.Count == 0) return Array.Empty<MlSpan>();

            var (tags, confs) = Tag(words);
            var length = tags.Length;
            var located = Locate(text, words.Take(length).ToList());
            var results = new List<MlSpan>();

            var i = 0;
            while (i < length)
            {
                if (!tags[i].StartsWith("B-", StringComparison.Ordinal)) { i++; continue; }
                var category = tags[i][2..];
                var j = i + 1;
                while (j < length && tags[j] == "I-" + category) j++;

                var spanLocs = located.Skip(i).Take(j - i).Where(l => l.HasValue).Select(l => l!.Value).ToList();
                if (spanLocs.Count > 0)
                {
                    var start = spanLocs.Min(l => l.start);
                    var end = spanLocs.Max(l => l.end);
                    var confidence = confs.Skip(i).Take(j - i).Average();
                    if (start < end && end <= text.Length)
                        results.Add(new MlSpan(start, end, text[start..end], LabelOf.GetValueOrDefault(category, category.ToLowerInvariant()), confidence));
                }
                i = j;
            }
            return results.Where(s => s.Confidence >= MinConfidence).ToList();
        }
        catch
        {
            return Array.Empty<MlSpan>();
        }
    }

    private static List<(int start, int end)?> Locate(string text, IReadOnlyList<string> words)
    {
        var spans = new List<(int, int)?>();
        var cursor = 0;
        foreach (var w in words)
        {
            var idx = text.IndexOf(w, cursor, StringComparison.Ordinal);
            if (idx < 0) idx = text.IndexOf(w, StringComparison.Ordinal);
            if (idx < 0) { spans.Add(null); continue; }
            spans.Add((idx, idx + w.Length));
            cursor = idx + w.Length;
        }
        return spans;
    }
}
