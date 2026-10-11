// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// Tokenizer and vocabulary of the Synthelion models. Both models were trained on the tokenization of Synthelion's
// core (structured values isolated first, then a letters/marks/digits word split), so inference MUST tokenize the
// same way - the patterns below are copied from the reference implementation, not from Caveman's compressor.
// ------------------------------------------------------------------------------
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using caveman.core;

namespace Caveman.ML.Text;

/// <summary>One token of the input text.</summary>
public sealed record MlToken(string Text, bool IsPunct, bool Protected);

public static class MlTokenizer
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(2);

    // Structured values (URLs, e-mails, IPv4/IPv6, ISO dates, dotted versions, numbers) kept whole.
    private static readonly Regex Protected = new(
        @"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+|" +
        @"[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|" +
        @"(?<!\d)(?:\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/\d{1,2})?|(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4})(?!\d)|" +
        @"(?<!\d)\d{4}-\d{2}-\d{2}(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?(?!\d)|" +
        @"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?!\d)|" +
        @"(?<!\d)v?\d+(?:\.\d+)+(?!\d)|" +
        @"(?<!\d)[+-]?(?:\d+(?:[.,]\d+)?)(?:[eE][+-]?\d+)?(?:[%‰°])?(?!\d)",
        RegexOptions.Compiled | RegexOptions.CultureInvariant, Timeout);

    private static readonly Regex WordSplit = new(
        @"[\p{L}\p{M}\p{N}_]+(?:'[\p{L}\p{M}\p{N}_]+)?|[^\w\s]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant, Timeout);

    private static readonly FunctionWordProvider CjkProvider = new();

    /// <summary>Splits <paramref name="text"/> into tokens exactly as the training pipeline did.</summary>
    public static List<MlToken> Tokenize(string text)
    {
        var tokens = new List<MlToken>();
        var cursor = 0;

        void AppendNormal(string segment)
        {
            if (segment.Length == 0) return;
            foreach (Match m in WordSplit.Matches(segment))
            {
                var t = m.Value;
                var isPunct = !(char.IsLetter(t[0]) || char.IsDigit(t[0]));
                if (!isPunct && t.Length > 1 && CavemanCjkSegmenter.IsHan(t[0]))
                {
                    foreach (var w in CavemanCjkSegmenter.SegmentHanRun(t, CjkProvider))
                        tokens.Add(new MlToken(w, false, false));
                    continue;
                }
                tokens.Add(new MlToken(t, isPunct, false));
            }
        }

        foreach (Match m in Protected.Matches(text))
        {
            if (m.Index > cursor) AppendNormal(text.Substring(cursor, m.Index - cursor));
            var value = m.Value;
            // Sentence punctuation belongs to the sentence, not to the protected value.
            var trailing = new StringBuilder();
            while (value.Length > 0 && ".,!?;:".IndexOf(value[^1]) >= 0)
            {
                trailing.Insert(0, value[^1]);
                value = value[..^1];
            }
            if (value.Length > 0) tokens.Add(new MlToken(value, false, true));
            if (trailing.Length > 0) AppendNormal(trailing.ToString());
            cursor = m.Index + m.Length;
        }
        if (cursor < text.Length) AppendNormal(text.Substring(cursor));
        return tokens;
    }

    /// <summary>
    /// Joins surviving tokens without inserting spaces inside Chinese/Japanese runs (written without spaces), keeping
    /// a separator between a CJK token and a Latin one.
    /// </summary>
    public static string Join(IReadOnlyList<string> items)
    {
        if (items.Count == 0) return "";
        var sb = new StringBuilder(items[0]);
        for (var i = 1; i < items.Count; i++)
        {
            var prev = items[i - 1]; var cur = items[i];
            if (!(prev.Length > 0 && cur.Length > 0 && IsCjk(prev[^1]) && IsCjk(cur[0]))) sb.Append(' ');
            sb.Append(cur);
        }
        return sb.ToString();
    }

    private static bool IsCjk(char c) =>
        (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x3040 && c <= 0x309F)
        || (c >= 0x30A0 && c <= 0x30FF) || (c >= 0xF900 && c <= 0xFAFF);
}

/// <summary>Word vocabulary with a deterministic char-n-gram hash fallback for out-of-vocabulary words.</summary>
public sealed class MlVocab
{
    public const int CharNgramBuckets = 8192;
    private readonly Dictionary<string, int> _word2id;

    public int Size => _word2id.Count;
    public int UnkId { get; }

    private MlVocab(Dictionary<string, int> word2id)
    {
        _word2id = word2id;
        UnkId = word2id.TryGetValue("<unk>", out var u) ? u : 1;
    }

    public static MlVocab Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        var i = 0;
        foreach (var w in doc.RootElement.GetProperty("words").EnumerateArray())
            map[w.GetString()!] = i++;
        return new MlVocab(map);
    }

    public bool TryGet(string word, out int id) => _word2id.TryGetValue(word, out id);

    /// <summary>Vocabulary id, or the char-n-gram bucket id offset by the vocabulary size.</summary>
    public int Encode(string word) => _word2id.TryGetValue(word, out var id) ? id : CharNgramHash(word) + Size;

    /// <summary>
    /// Deterministic hash of the word's character trigrams (the out-of-vocabulary fallback): md5 of the
    /// "|"-joined trigrams of " word ", reduced modulo 8192 - bit-identical to the training pipeline.
    /// </summary>
    public static int CharNgramHash(string word, int buckets = CharNgramBuckets)
    {
        var padded = " " + word + " ";
        var combined = padded.Length < 5
            ? padded
            : string.Join("|", Enumerable.Range(0, padded.Length - 2).Select(i => padded.Substring(i, 3)));
        var digest = MD5.HashData(Encoding.UTF8.GetBytes(combined));
        // int(hexdigest, 16) % buckets: the digest is a big-endian 128-bit number, so reduce byte by byte.
        long acc = 0;
        foreach (var b in digest) acc = (acc * 256 + b) % buckets;
        return (int)acc;
    }
}
