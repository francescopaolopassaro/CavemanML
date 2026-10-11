// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// Local model management. The two models live on Hugging Face (digitalsolutionsai/Synthelion and
// digitalsolutionsai/PrivacyGuardML); ModelManager downloads them once into ~/.caveman/ml_models/<name>/, records
// a manifest (repository, revision, size and SHA-256 of every file), and from then on everything runs offline:
//
//   status  - what is installed, where, how big, which revision
//   install - download what is missing (resumable per file, atomic, SHA-256 checked against the Hub's LFS digest)
//   verify  - re-hash the files on disk against the manifest
//   update  - is a newer revision available on the Hub?
//   remove  - delete the local copy
// ------------------------------------------------------------------------------
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Caveman.ML.Model;

/// <summary>One file of an installed model, as recorded in the manifest.</summary>
public sealed record ModelFile(string Name, long Size, string? Sha256);

/// <summary>The record written next to the weights (<c>.caveman-model.json</c>).</summary>
public sealed record ModelManifest(string Repo, string Revision, DateTimeOffset InstalledAt, IReadOnlyList<ModelFile> Files);

/// <summary>What <see cref="ModelManager.GetStatus"/> reports about one model.</summary>
public sealed record ModelStatus(
    MlModelKind Kind, string Repo, bool Installed, string? Directory, long SizeBytes, string? Revision,
    DateTimeOffset? InstalledAt, IReadOnlyList<string> Files);

/// <summary>Outcome of <see cref="ModelManager.VerifyAsync"/>.</summary>
public sealed record VerifyResult(bool Ok, IReadOnlyList<string> Problems);

/// <summary>Outcome of <see cref="ModelManager.CheckUpdateAsync"/>.</summary>
public sealed record UpdateInfo(bool UpdateAvailable, string? InstalledRevision, string? LatestRevision);

public static class ModelManager
{
    private const string ManifestName = ".caveman-model.json";
    private const string HubApi = "https://huggingface.co/api/models/";
    private const string HubResolve = "https://huggingface.co/";

    private sealed record Spec(string Folder, string Repo, string[] Files, string[] WeightFiles);

    private static Spec SpecOf(MlModelKind kind) => kind switch
    {
        // SynthelionML ships safetensors (and ONNX/bin, which are not needed); PrivacyGuardML ships only the PyTorch .bin.
        MlModelKind.SynthelionML => new("synthelionml", "digitalsolutionsai/Synthelion",
            new[] { "config.json", "vocab.json", "model.safetensors" }, new[] { "model.safetensors", "model.bin" }),
        _ => new("privacyguardml", "digitalsolutionsai/PrivacyGuardML",
            new[] { "config.json", "vocab.json", "model.bin" }, new[] { "model.safetensors", "model.bin" }),
    };

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };

    public static IReadOnlyList<MlModelKind> Kinds { get; } = new[] { MlModelKind.SynthelionML, MlModelKind.PrivacyGuardML };

    /// <summary>The Hugging Face repository a model is downloaded from.</summary>
    public static string RepoOf(MlModelKind kind) => SpecOf(kind).Repo;

    /// <summary>Where a model is installed by <see cref="InstallAsync"/> (<c>~/.caveman/ml_models/&lt;name&gt;</c>, or under CAVEMAN_ML_MODELS_DIR).</summary>
    public static string InstallDirectory(MlModelKind kind)
    {
        var root = Environment.GetEnvironmentVariable("CAVEMAN_ML_MODELS_DIR");
        return Path.Combine(string.IsNullOrEmpty(root) ? ModelStore.DefaultRoot() : root, SpecOf(kind).Folder);
    }

    private static HttpClient NewClient()
    {
        var http = new HttpClient { Timeout = TimeSpan.FromMinutes(15) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("CavemanML/1.0");
        var token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token)) http.DefaultRequestHeaders.Authorization = new("Bearer", token);
        return http;
    }

    // ───────────────────────────── status ─────────────────────────────

    public static IReadOnlyList<ModelStatus> GetStatus() => Kinds.Select(GetStatus).ToList();

    public static ModelStatus GetStatus(MlModelKind kind)
    {
        var spec = SpecOf(kind);
        var dir = ModelStore.Resolve(kind);
        if (dir is null) return new ModelStatus(kind, spec.Repo, false, null, 0, null, null, Array.Empty<string>());

        var files = Directory.EnumerateFiles(dir).Where(f => Path.GetFileName(f) != ManifestName).ToList();
        var manifest = ReadManifest(dir);
        return new ModelStatus(kind, spec.Repo, true, dir, files.Sum(f => new FileInfo(f).Length), manifest?.Revision,
            manifest?.InstalledAt, files.Select(Path.GetFileName).OrderBy(n => n, StringComparer.Ordinal).ToList()!);
    }

    private static ModelManifest? ReadManifest(string dir)
    {
        try
        {
            var path = Path.Combine(dir, ManifestName);
            return File.Exists(path) ? JsonSerializer.Deserialize<ModelManifest>(File.ReadAllText(path), Json) : null;
        }
        catch { return null; }
    }

    // ───────────────────────────── install ─────────────────────────────

    private sealed record HubFile(string Path, long Size, string? Sha256);

    private static async Task<(string Revision, Dictionary<string, HubFile> Files)> HubInfoAsync(HttpClient http, string repo, CancellationToken ct)
    {
        using var meta = JsonDocument.Parse(await http.GetStringAsync(HubApi + repo, ct).ConfigureAwait(false));
        var revision = meta.RootElement.GetProperty("sha").GetString() ?? "";
        using var tree = JsonDocument.Parse(await http.GetStringAsync($"{HubApi}{repo}/tree/main", ct).ConfigureAwait(false));
        var files = new Dictionary<string, HubFile>(StringComparer.Ordinal);
        foreach (var f in tree.RootElement.EnumerateArray())
        {
            if (f.GetProperty("type").GetString() != "file") continue;
            var path = f.GetProperty("path").GetString()!;
            string? sha = null;
            if (f.TryGetProperty("lfs", out var lfs) && lfs.TryGetProperty("oid", out var oid)) sha = oid.GetString();
            files[path] = new HubFile(path, f.GetProperty("size").GetInt64(), sha);
        }
        return (revision, files);
    }

    /// <summary>
    /// Downloads a model into its install directory and writes the manifest. Files already present with the right size are kept
    /// (unless <paramref name="force"/>); every download goes to a temporary file, is SHA-256 checked against the Hub's digest,
    /// and is moved into place atomically - a half-downloaded or corrupt file is never mistaken for a model.
    /// </summary>
    /// <exception cref="HttpRequestException">The Hub could not be reached.</exception>
    /// <exception cref="InvalidDataException">A downloaded file does not match the Hub's checksum.</exception>
    public static async Task<ModelStatus> InstallAsync(MlModelKind kind, bool force = false,
        IProgress<(string File, long Done, long Total)>? progress = null, CancellationToken ct = default)
    {
        var spec = SpecOf(kind);
        var target = InstallDirectory(kind);
        Directory.CreateDirectory(target);

        using var http = NewClient();
        var (revision, hub) = await HubInfoAsync(http, spec.Repo, ct).ConfigureAwait(false);
        var recorded = new List<ModelFile>();

        foreach (var name in spec.Files)
        {
            if (!hub.TryGetValue(name, out var info))
                throw new InvalidDataException($"{spec.Repo} has no file '{name}' at revision {revision}");
            var dest = Path.Combine(target, name);

            if (!force && File.Exists(dest) && new FileInfo(dest).Length == info.Size)
            {
                recorded.Add(new ModelFile(name, info.Size, info.Sha256));
                continue;
            }

            var tmp = dest + ".part";
            using (var resp = await http.GetAsync($"{HubResolve}{spec.Repo}/resolve/{revision}/{name}", HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false))
            {
                resp.EnsureSuccessStatusCode();
                await using var src = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                await using var fs = File.Create(tmp);
                using var sha = SHA256.Create();
                var buffer = new byte[1 << 16];
                long done = 0, total = resp.Content.Headers.ContentLength ?? info.Size;
                int read;
                while ((read = await src.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
                {
                    await fs.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    sha.TransformBlock(buffer, 0, read, null, 0);
                    done += read;
                    progress?.Report((name, done, total));
                }
                sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                var hex = Convert.ToHexString(sha.Hash!).ToLowerInvariant();
                await fs.FlushAsync(ct).ConfigureAwait(false);
                if (info.Sha256 is not null && !hex.Equals(info.Sha256, StringComparison.OrdinalIgnoreCase))
                {
                    fs.Close();
                    File.Delete(tmp);
                    throw new InvalidDataException($"checksum mismatch for {name}: expected {info.Sha256}, got {hex}");
                }
                info = info with { Sha256 = info.Sha256 ?? hex };
            }
            File.Move(tmp, dest, overwrite: true);
            recorded.Add(new ModelFile(name, info.Size, info.Sha256));
        }

        File.WriteAllText(Path.Combine(target, ManifestName),
            JsonSerializer.Serialize(new ModelManifest(spec.Repo, revision, DateTimeOffset.UtcNow, recorded), Json));
        return GetStatus(kind);
    }

    // ───────────────────────────── verify / update / remove ─────────────────────────────

    /// <summary>Re-hashes the installed files against the manifest. A model installed without a manifest is checked for completeness only.</summary>
    public static async Task<VerifyResult> VerifyAsync(MlModelKind kind, CancellationToken ct = default)
    {
        var dir = ModelStore.Resolve(kind);
        if (dir is null) return new VerifyResult(false, new[] { "not installed" });
        var problems = new List<string>();
        var manifest = ReadManifest(dir);

        if (manifest is null)
        {
            foreach (var f in SpecOf(kind).Files)
                if (!File.Exists(Path.Combine(dir, f)) && !(f.StartsWith("model.", StringComparison.Ordinal) && SpecOf(kind).WeightFiles.Any(w => File.Exists(Path.Combine(dir, w)))))
                    problems.Add($"missing {f}");
            return new VerifyResult(problems.Count == 0, problems);
        }

        foreach (var f in manifest.Files)
        {
            var path = Path.Combine(dir, f.Name);
            if (!File.Exists(path)) { problems.Add($"missing {f.Name}"); continue; }
            if (new FileInfo(path).Length != f.Size) { problems.Add($"{f.Name}: size {new FileInfo(path).Length} != {f.Size}"); continue; }
            if (f.Sha256 is null) continue;
            await using var fs = File.OpenRead(path);
            var hex = Convert.ToHexString(await SHA256.HashDataAsync(fs, ct).ConfigureAwait(false)).ToLowerInvariant();
            if (!hex.Equals(f.Sha256, StringComparison.OrdinalIgnoreCase)) problems.Add($"{f.Name}: SHA-256 mismatch");
        }
        return new VerifyResult(problems.Count == 0, problems);
    }

    /// <summary>Asks the Hub whether a newer revision of the model exists.</summary>
    public static async Task<UpdateInfo> CheckUpdateAsync(MlModelKind kind, CancellationToken ct = default)
    {
        var dir = ModelStore.Resolve(kind);
        var installed = dir is null ? null : ReadManifest(dir)?.Revision;
        using var http = NewClient();
        using var meta = JsonDocument.Parse(await http.GetStringAsync(HubApi + SpecOf(kind).Repo, ct).ConfigureAwait(false));
        var latest = meta.RootElement.GetProperty("sha").GetString();
        return new UpdateInfo(installed is not null && latest is not null && installed != latest, installed, latest);
    }

    /// <summary>Deletes the local copy of a model installed by this manager. Returns false when nothing was installed there.</summary>
    public static bool Remove(MlModelKind kind)
    {
        var dir = InstallDirectory(kind);
        if (!Directory.Exists(dir)) return false;
        Directory.Delete(dir, recursive: true);
        return true;
    }
}
