using System.Text.Json;
using Caveman.ML;
using Caveman.ML.Model;
using Caveman.ML.Text;
using Caveman.ML.Weights;
using Caveman.PrivacyGuard;
using NUnit.Framework;

namespace CavemanML.Tests;

/// <summary>
/// Parity with the PyTorch reference (Synthelion 1.2.5). tools/make_reference.py ran the reference models on a fixed
/// corpus and stored per-word scores, tags, spans and compressed text; the C# engine must reproduce them.
/// The tests need the models on disk (~/.caveman/ml_models or CAVEMAN_ML_MODELS_DIR) and are ignored otherwise.
/// </summary>
[TestFixture]
public class ParityTests
{
    private JsonElement _ref;
    private SynthelionMLCompressor? _synth;
    private PrivacyGuardMLDetector? _pg;

    [OneTimeSetUp]
    public void Setup()
    {
        _ref = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "reference.json"))).RootElement;
        _synth = SynthelionMLCompressor.Load();
        _pg = PrivacyGuardMLDetector.Load(minConfidence: 0.0);
    }

    private static string[] Strings(JsonElement e) => e.EnumerateArray().Select(x => x.GetString()!).ToArray();
    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    // ── weight formats ───────────────────────────────────────────────────

    [Test]
    public void TorchCheckpoint_And_SafeTensors_AreIdentical()
    {
        var dir = ModelStore.Resolve(MlModelKind.SynthelionML);
        if (dir is null || !File.Exists(Path.Combine(dir, "model.safetensors")) || !File.Exists(Path.Combine(dir, "model.bin")))
            Assert.Ignore("SynthelionML (both formats) not installed");
        var a = TensorFiles.LoadSafeTensors(Path.Combine(dir!, "model.safetensors"));
        var b = TensorFiles.LoadTorchCheckpoint(Path.Combine(dir!, "model.bin"));
        Assert.That(b.Keys.OrderBy(k => k), Is.EqualTo(a.Keys.OrderBy(k => k)));
        foreach (var (name, t) in a)
        {
            Assert.That(b[name].Shape, Is.EqualTo(t.Shape), name);
            Assert.That(b[name].Data, Is.EqualTo(t.Data), name);
        }
    }

    // ── SynthelionML ─────────────────────────────────────────────────────

    [Test]
    public void SynthelionML_Scores_MatchTheReference()
    {
        if (_synth is null) Assert.Ignore("SynthelionML not installed");
        var worst = 0.0;
        foreach (var c in _ref.GetProperty("synthelionml").GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var words = MlTokenizer.Tokenize(text).Where(t => !t.IsPunct).Select(t => t.Text).ToArray();
            Assert.That(words, Is.EqualTo(Strings(c.GetProperty("words"))), $"tokenization: {text}");

            var expected = Doubles(c.GetProperty("scores"));
            // the engine is checked in isolation: same stop-word flags as the reference run (word lists differ slightly between the ports)
            var stop = c.GetProperty("stop").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
            var map = words.Zip(stop).GroupBy(x => x.First).ToDictionary(g => g.Key, g => g.First().Second);
            var actual = _synth!.ScoreWords(words, w => map[w]);
            Assert.That(actual.Count, Is.EqualTo(expected.Length));
            for (var i = 0; i < expected.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-3), $"'{words[i]}' in: {text}");
            }
        }
        TestContext.Out.WriteLine($"max |keep-probability difference| vs PyTorch: {worst:E2}");
    }

    [Test]
    public void SynthelionML_EndToEndScores_StayCloseWithCavemansWordLists()
    {
        // Caveman and Synthelion curate their stop-word lists independently, and "is stop word" is one of the model's inputs, so
        // a word on one list but not the other can move its neighbours' scores. On average the two stay close.
        if (_synth is null) Assert.Ignore("SynthelionML not installed");
        double total = 0; var n = 0;
        foreach (var c in _ref.GetProperty("synthelionml").GetProperty("cases").EnumerateArray())
        {
            var words = Strings(c.GetProperty("words"));
            var expected = Doubles(c.GetProperty("scores"));
            var actual = _synth!.ScoreWords(words, c.GetProperty("iso3").GetString()!);
            for (var i = 0; i < expected.Length; i++) { total += Math.Abs(actual[i] - expected[i]); n++; }
        }
        Assert.That(total / n, Is.LessThan(0.1), "mean |difference| of keep probabilities vs the reference");
    }

    [Test]
    public void SynthelionML_CompressedText_MatchesTheReference()
    {
        if (_synth is null) Assert.Ignore("SynthelionML not installed");
        foreach (var c in _ref.GetProperty("synthelionml").GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var iso3 = c.GetProperty("iso3").GetString()!;
            Assert.That(_synth!.Compress(text, iso3), Is.EqualTo(c.GetProperty("compressed").GetString()), text);
            Assert.That(_synth.Compress(text, iso3, budget: 6), Is.EqualTo(c.GetProperty("compressed_budget_6").GetString()), "budget: " + text);
        }
    }

    [Test]
    public void SynthelionML_KeepsValuesNegationsAndNames_AndNeverEmptiesASentence()
    {
        if (_synth is null) Assert.Ignore("SynthelionML not installed");
        var r = _synth!.Compress("Please make sure that the server at 192.168.1.10:8080 is not reachable before 2026-10-11 in Rome.", "eng");
        Assert.That(r, Does.Contain("192.168.1.10:8080").And.Contain("2026-10-11").And.Contain("not").And.Contain("Rome"));
        Assert.That(_synth.Compress("Go.", "eng"), Is.Not.Empty);
    }

    [Test]
    public void SynthelionML_HonoursAMinimumCompressionRatio()
    {
        if (_synth is null) Assert.Ignore("SynthelionML not installed");
        const string text = "I would like to know if it is possible to receive some information about the cheap restaurants that are open near the main station.";
        var light = _synth!.CompressResult(text, "eng", minCompression: 0.3);
        var heavy = _synth.CompressResult(text, "eng", minCompression: 0.8);
        Assert.That(heavy.CompressedTokens, Is.LessThan(light.CompressedTokens));
        Assert.That(heavy.CompressedTokens, Is.GreaterThanOrEqualTo(1));
    }

    // ── PrivacyGuardML ───────────────────────────────────────────────────

    [Test]
    public void PrivacyGuardML_Tags_MatchTheReference()
    {
        if (_pg is null) Assert.Ignore("PrivacyGuardML not installed");
        foreach (var c in _ref.GetProperty("privacyguardml").GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var words = MlTokenizer.Tokenize(text).Where(t => !t.IsPunct).Select(t => t.Text).ToArray();
            Assert.That(words, Is.EqualTo(Strings(c.GetProperty("words"))), text);

            var (tags, confs) = _pg!.Tag(words);
            Assert.That(tags, Is.EqualTo(Strings(c.GetProperty("tags"))), text);
            var expected = Doubles(c.GetProperty("confidences"));
            for (var i = 0; i < expected.Length; i++)
                Assert.That(confs[i], Is.EqualTo(expected[i]).Within(2e-3), $"'{words[i]}' in: {text}");
        }
    }

    [Test]
    public void PrivacyGuardML_Spans_MatchTheReference()
    {
        if (_pg is null) Assert.Ignore("PrivacyGuardML not installed");
        foreach (var c in _ref.GetProperty("privacyguardml").GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var expected = c.GetProperty("spans").EnumerateArray().ToList();
            var actual = _pg!.Detect(text);
            Assert.That(actual.Count, Is.EqualTo(expected.Count), text);
            for (var i = 0; i < expected.Count; i++)
            {
                Assert.That(actual[i].Start, Is.EqualTo(expected[i].GetProperty("start").GetInt32()), text);
                Assert.That(actual[i].End, Is.EqualTo(expected[i].GetProperty("end").GetInt32()), text);
                Assert.That(actual[i].Label, Is.EqualTo(expected[i].GetProperty("label").GetString()), text);
                Assert.That(actual[i].Confidence, Is.EqualTo(expected[i].GetProperty("confidence").GetDouble()).Within(2e-3), text);
            }
        }
    }

    [Test]
    public void PrivacyGuardML_FindsNothingInHarmlessText()
    {
        if (_pg is null) Assert.Ignore("PrivacyGuardML not installed");
        var spans = new PrivacyGuardMLDetector[] { PrivacyGuardMLDetector.FromDirectory(_pg!.ModelDirectory) }[0]
            .Detect("Meeting at the office next Tuesday about the budget, nothing personal here.");
        Assert.That(spans, Is.Empty);
    }

    // ── integration with Caveman.PrivacyGuard ────────────────────────────

    [Test]
    public void AnalyzerWithML_ConfirmsABareValue_ButNeverOverridesAChecksum()
    {
        if (_pg is null) Assert.Ignore("PrivacyGuardML not installed");
        using var plain = new PrivacyAnalyzer();
        using var withMl = new PrivacyAnalyzer { MlDetector = PrivacyGuardMLDetector.FromDirectory(_pg!.ModelDirectory) };

        // an invalid Italian codice fiscale: the checksum vetoes it with or without ML
        const string badCf = "il codice RSSMRA85T10H501Z e' stato inviato";
        Assert.That(withMl.Analyze(badCf).DetectedCategories, Does.Not.Contain("Italian Tax Code (CF)"));

        // a valid IBAN is structurally anchored: detected by both, ML adds nothing and breaks nothing
        const string iban = "IBAN IT60X0542811101000000123456";
        Assert.That(plain.Analyze(iban).DetectedCategories, Does.Contain("IBAN"));
        Assert.That(withMl.Analyze(iban).DetectedCategories, Does.Contain("IBAN"));

        // the ML tier is opt-in: without a detector nothing is ML-assisted
        Assert.That(plain.Analyze("tel 3331234567 per favore").MlAssistedCount, Is.EqualTo(0));
    }
}
