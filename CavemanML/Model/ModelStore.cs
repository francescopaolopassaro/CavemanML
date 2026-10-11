// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// Where the models live. Resolution order (first match wins):
//   1. an explicit directory passed by the caller
//   2. $CAVEMAN_ML_MODEL_<NAME>  (a single model directory)
//   3. $CAVEMAN_ML_MODELS_DIR/<name>
//   4. ~/.caveman/ml_models/<name>
//   5. ~/.synthelion/ml_models/<name>   (shared with the Python port)
//   6. one-time download from Hugging Face into ~/.caveman/ml_models/<name>
// Inference itself is fully offline; the download is a convenience, and is skipped when the files are already there.
// ------------------------------------------------------------------------------
namespace Caveman.ML.Model;

/// <summary>The two models shipped by the Caveman ML package.</summary>
public enum MlModelKind
{
    /// <summary>Learned per-token keep/drop prompt compressor.</summary>
    SynthelionML,
    /// <summary>Learned PII confirmation model (BIO tagging).</summary>
    PrivacyGuardML,
}

public static class ModelStore
{
    private sealed record Spec(string Folder, string Repo, string[] Files, string[] WeightFiles);

    private static Spec SpecOf(MlModelKind kind) => kind switch
    {
        MlModelKind.SynthelionML => new("synthelionml", "digitalsolutionsai/Synthelion",
            new[] { "config.json", "vocab.json", "model.safetensors" }, new[] { "model.safetensors", "model.bin" }),
        _ => new("privacyguardml", "digitalsolutionsai/PrivacyGuardML",
            new[] { "config.json", "vocab.json", "model.bin" }, new[] { "model.safetensors", "model.bin" }),
    };

    public static string DefaultRoot() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".caveman", "ml_models");

    private static bool LooksLikeModelDir(string dir, Spec spec) =>
        File.Exists(Path.Combine(dir, "config.json")) && File.Exists(Path.Combine(dir, "vocab.json"))
        && spec.WeightFiles.Any(f => File.Exists(Path.Combine(dir, f)));

    /// <summary>Finds an installed model directory, or <c>null</c> when none is present locally.</summary>
    public static string? Resolve(MlModelKind kind, string? explicitDir = null)
    {
        var spec = SpecOf(kind);
        if (explicitDir is not null) return LooksLikeModelDir(explicitDir, spec) ? explicitDir : null;

        var env = Environment.GetEnvironmentVariable("CAVEMAN_ML_MODEL_" + spec.Folder.ToUpperInvariant());
        if (!string.IsNullOrEmpty(env) && LooksLikeModelDir(env, spec)) return env;

        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var roots = new List<string>();
        var rootEnv = Environment.GetEnvironmentVariable("CAVEMAN_ML_MODELS_DIR");
        if (!string.IsNullOrEmpty(rootEnv)) roots.Add(rootEnv);
        roots.Add(DefaultRoot());
        roots.Add(Path.Combine(home, ".synthelion", "ml_models"));
        foreach (var root in roots)
        {
            var candidate = Path.Combine(root, spec.Folder);
            if (LooksLikeModelDir(candidate, spec)) return candidate;
        }
        return null;
    }

    /// <summary>
    /// Returns the model directory, downloading it from Hugging Face on first use when it is not installed.
    /// Returns <c>null</c> (never throws) when the model is absent and cannot be downloaded.
    /// </summary>
    public static async Task<string?> EnsureAsync(MlModelKind kind, string? explicitDir = null,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var found = Resolve(kind, explicitDir);
        if (found is not null || explicitDir is not null) return found;

        try
        {
            IProgress<(string File, long Done, long Total)>? p = progress is null ? null
                : new Progress<(string File, long Done, long Total)>(x => progress.Report($"downloading {x.File} {x.Done / 1048576} / {x.Total / 1048576} MB"));
            var status = await ModelManager.InstallAsync(kind, force: false, p, ct).ConfigureAwait(false);
            return status.Directory;
        }
        catch (Exception) when (!ct.IsCancellationRequested)
        {
            return null;
        }
    }
}
