using Caveman.ML;
using Caveman.ML.Model;
using NUnit.Framework;

namespace CavemanML.Tests;

/// <summary>
/// End-to-end test of the model manager against the real Hugging Face repositories (downloads ~70 MB into a temporary
/// folder). Marked Explicit so a normal test run stays offline: run it with
/// <c>dotnet test --filter "TestCategory=Network"</c>.
/// </summary>
[TestFixture, Category("Network"), Explicit("downloads the models from Hugging Face")]
public class ModelManagerTests
{
    private string _root = "";
    private string? _previous;

    [OneTimeSetUp]
    public void Setup()
    {
        _root = Path.Combine(Path.GetTempPath(), "cavemanml-models-" + Guid.NewGuid().ToString("N"));
        _previous = Environment.GetEnvironmentVariable("CAVEMAN_ML_MODELS_DIR");
        Environment.SetEnvironmentVariable("CAVEMAN_ML_MODELS_DIR", _root);
    }

    [OneTimeTearDown]
    public void TearDown()
    {
        Environment.SetEnvironmentVariable("CAVEMAN_ML_MODELS_DIR", _previous);
        try { Directory.Delete(_root, true); } catch { }
    }

    [Test, Order(1)]
    public async Task Install_Downloads_Verifies_AndWritesAManifest()
    {
        foreach (var kind in ModelManager.Kinds)
        {
            var progress = new List<string>();
            var status = await ModelManager.InstallAsync(kind, progress: new Progress<(string File, long Done, long Total)>(p => progress.Add(p.File)));
            Assert.That(status.Installed, Is.True, kind.ToString());
            Assert.That(status.Directory, Does.StartWith(_root));
            Assert.That(status.Revision, Is.Not.Null.And.Not.Empty);
            Assert.That(status.SizeBytes, Is.GreaterThan(20_000_000));
            Assert.That(progress, Is.Not.Empty);
            Assert.That(File.Exists(Path.Combine(status.Directory!, ".caveman-model.json")), Is.True);

            var verify = await ModelManager.VerifyAsync(kind);
            Assert.That(verify.Ok, Is.True, string.Join("; ", verify.Problems));
        }
    }

    [Test, Order(2)]
    public async Task InstalledModels_AreUsable_AndASecondInstallIsANoOp()
    {
        var synth = await SynthelionMLCompressor.LoadAsync(download: false);
        var pg = await PrivacyGuardMLDetector.LoadAsync(download: false);
        Assert.That(synth, Is.Not.Null);
        Assert.That(pg, Is.Not.Null);
        Assert.That(synth!.Compress("I would like to know if it is possible to receive some information.", "eng"), Is.Not.Empty);

        var stamp = File.GetLastWriteTimeUtc(Path.Combine(ModelManager.InstallDirectory(MlModelKind.PrivacyGuardML), "model.bin"));
        await ModelManager.InstallAsync(MlModelKind.PrivacyGuardML);
        Assert.That(File.GetLastWriteTimeUtc(Path.Combine(ModelManager.InstallDirectory(MlModelKind.PrivacyGuardML), "model.bin")), Is.EqualTo(stamp), "files already present are kept");
    }

    [Test, Order(3)]
    public async Task Verify_DetectsACorruptedFile()
    {
        var path = Path.Combine(ModelManager.InstallDirectory(MlModelKind.PrivacyGuardML), "vocab.json");
        var original = await File.ReadAllBytesAsync(path);
        var corrupted = (byte[])original.Clone();
        corrupted[^2] ^= 0x01;                         // same size, different content
        await File.WriteAllBytesAsync(path, corrupted);
        try
        {
            var verify = await ModelManager.VerifyAsync(MlModelKind.PrivacyGuardML);
            Assert.That(verify.Ok, Is.True, "small config files carry no LFS digest; size is the check");
        }
        finally { await File.WriteAllBytesAsync(path, original); }

        var weights = Path.Combine(ModelManager.InstallDirectory(MlModelKind.PrivacyGuardML), "model.bin");
        var w = await File.ReadAllBytesAsync(weights);
        w[1000] ^= 0xFF;
        await File.WriteAllBytesAsync(weights, w);
        try
        {
            var verify = await ModelManager.VerifyAsync(MlModelKind.PrivacyGuardML);
            Assert.That(verify.Ok, Is.False);
            Assert.That(verify.Problems.Any(p => p.Contains("SHA-256")), Is.True, string.Join("; ", verify.Problems));
        }
        finally { w[1000] ^= 0xFF; await File.WriteAllBytesAsync(weights, w); }
    }

    [Test, Order(4)]
    public async Task CheckUpdate_ReportsTheInstalledRevision()
    {
        var info = await ModelManager.CheckUpdateAsync(MlModelKind.SynthelionML);
        Assert.That(info.InstalledRevision, Is.Not.Null);
        Assert.That(info.LatestRevision, Is.Not.Null);
        Assert.That(info.UpdateAvailable, Is.False);
    }

    [Test, Order(5)]
    public void Remove_DeletesTheLocalCopy()
    {
        Assert.That(ModelManager.Remove(MlModelKind.SynthelionML), Is.True);
        Assert.That(Directory.Exists(ModelManager.InstallDirectory(MlModelKind.SynthelionML)), Is.False);
        Assert.That(ModelManager.Remove(MlModelKind.SynthelionML), Is.False, "nothing left to remove");
    }
}
