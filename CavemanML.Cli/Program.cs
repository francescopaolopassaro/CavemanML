// caveman-ml - manage and run the CavemanML models.
//
//   caveman-ml models                         status of every model
//   caveman-ml install [synthelionml|privacyguardml|all] [--force]
//   caveman-ml verify  [name|all]             re-hash installed files against the manifest
//   caveman-ml update  [name|all]             is a newer revision on Hugging Face?  (--install to fetch it)
//   caveman-ml remove  <name|all>
//   caveman-ml compress "text" [--lang ita] [--ratio 0.7] [--budget N]
//   caveman-ml pii "text"                     PII entities found by PrivacyGuardML
using Caveman.ML;
using Caveman.ML.Model;

static string Size(long bytes) => bytes >= 1 << 20 ? $"{bytes / 1048576.0:F1} MB" : $"{bytes / 1024.0:F0} KB";

static IEnumerable<MlModelKind> Pick(string? name)
{
    if (name is null or "all") return ModelManager.Kinds;
    if (name.Equals("synthelionml", StringComparison.OrdinalIgnoreCase) || name.Equals("synthelion", StringComparison.OrdinalIgnoreCase)) return new[] { MlModelKind.SynthelionML };
    if (name.Equals("privacyguardml", StringComparison.OrdinalIgnoreCase) || name.Equals("privacyguard", StringComparison.OrdinalIgnoreCase)) return new[] { MlModelKind.PrivacyGuardML };
    throw new ArgumentException($"unknown model '{name}' (use synthelionml, privacyguardml or all)");
}

static string? Opt(string[] args, string name)
{
    var i = Array.FindIndex(args, a => a == name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

static void PrintStatus(ModelStatus s)
{
    Console.WriteLine($"{s.Kind,-15} {(s.Installed ? "installed" : "not installed")}");
    Console.WriteLine($"  repository : https://huggingface.co/{s.Repo}");
    if (!s.Installed) { Console.WriteLine($"  install    : caveman-ml install {s.Kind.ToString().ToLowerInvariant()}"); return; }
    Console.WriteLine($"  directory  : {s.Directory}");
    Console.WriteLine($"  size       : {Size(s.SizeBytes)}  ({string.Join(", ", s.Files)})");
    Console.WriteLine($"  revision   : {s.Revision ?? "(installed without a manifest)"}{(s.InstalledAt is { } t ? $"  installed {t:yyyy-MM-dd}" : "")}");
}

async Task<int> InstallAsync(IEnumerable<MlModelKind> kinds, bool force)
{
    foreach (var kind in kinds)
    {
        Console.WriteLine($"Installing {kind} from huggingface.co/{ModelManager.RepoOf(kind)} ...");
        string? last = null;
        var progress = new Progress<(string File, long Done, long Total)>(p =>
        {
            if (last != p.File) { if (last is not null) Console.WriteLine(); last = p.File; }
            Console.Write($"\r  {p.File,-20} {Size(p.Done),9} / {Size(p.Total)}   ");
        });
        var status = await ModelManager.InstallAsync(kind, force, progress);
        if (last is not null) Console.WriteLine();
        PrintStatus(status);
    }
    return 0;
}

try
{
    var cmd = args.FirstOrDefault()?.ToLowerInvariant();
    switch (cmd)
    {
        case null or "help" or "--help" or "-h":
            Console.WriteLine("caveman-ml - manage and run the CavemanML models\n\n" +
                "  models                              status of every model\n" +
                "  install [name|all] [--force]        download from Hugging Face (SHA-256 checked)\n" +
                "  verify  [name|all]                  re-hash installed files against the manifest\n" +
                "  update  [name|all] [--install]      check (and fetch) a newer revision\n" +
                "  remove  <name|all>                  delete the local copy\n" +
                "  compress \"text\" [--lang iso3] [--ratio 0.7] [--budget N]\n" +
                "  pii \"text\"                          PII entities found by PrivacyGuardML\n\n" +
                "models: synthelionml | privacyguardml | all     (HF_TOKEN is honoured for private mirrors)");
            return 0;

        case "models" or "status" or "list":
            foreach (var s in ModelManager.GetStatus()) { PrintStatus(s); Console.WriteLine(); }
            return 0;

        case "install":
            return await InstallAsync(Pick(args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"))), args.Contains("--force"));

        case "verify":
        {
            var bad = 0;
            foreach (var kind in Pick(args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"))))
            {
                var r = await ModelManager.VerifyAsync(kind);
                Console.WriteLine($"{kind,-15} {(r.Ok ? "OK" : "FAILED")}");
                foreach (var p in r.Problems) Console.WriteLine($"  - {p}");
                if (!r.Ok) bad++;
            }
            return bad == 0 ? 0 : 1;
        }

        case "update":
        {
            var fetch = args.Contains("--install");
            foreach (var kind in Pick(args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"))))
            {
                var u = await ModelManager.CheckUpdateAsync(kind);
                Console.WriteLine($"{kind,-15} installed {u.InstalledRevision?[..Math.Min(8, u.InstalledRevision.Length)] ?? "-"}  latest {u.LatestRevision?[..Math.Min(8, u.LatestRevision.Length)] ?? "-"}  {(u.UpdateAvailable ? "UPDATE AVAILABLE" : "up to date")}");
                if (u.UpdateAvailable && fetch) await InstallAsync(new[] { kind }, force: true);
            }
            return 0;
        }

        case "remove":
        {
            var name = args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"));
            if (name is null) { Console.Error.WriteLine("remove needs a model name (or 'all')"); return 2; }
            foreach (var kind in Pick(name))
                Console.WriteLine($"{kind,-15} {(ModelManager.Remove(kind) ? "removed" : "nothing to remove")}");
            return 0;
        }

        case "compress":
        {
            var text = args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"));
            if (text is null) { Console.Error.WriteLine("usage: caveman-ml compress \"text\""); return 2; }
            var ml = await SynthelionMLCompressor.LoadAsync(progress: new Progress<string>(Console.Error.WriteLine));
            if (ml is null) { Console.Error.WriteLine("SynthelionML is not installed and could not be downloaded."); return 1; }
            var ratio = double.TryParse(Opt(args, "--ratio"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var r) ? r : (double?)null;
            var budget = int.TryParse(Opt(args, "--budget"), out var b) ? b : (int?)null;
            var res = ml.CompressResult(text, Opt(args, "--lang"), budget, ratio);
            Console.WriteLine(res.CompressedText);
            Console.Error.WriteLine($"[{res.OriginalTokens} -> {res.CompressedTokens} words, {res.EfficiencyPercentage:F0}% saved]");
            return 0;
        }

        case "pii":
        {
            var text = args.Skip(1).FirstOrDefault(a => !a.StartsWith("--"));
            if (text is null) { Console.Error.WriteLine("usage: caveman-ml pii \"text\""); return 2; }
            var pg = await PrivacyGuardMLDetector.LoadAsync(minConfidence: 0.0, progress: new Progress<string>(Console.Error.WriteLine));
            if (pg is null) { Console.Error.WriteLine("PrivacyGuardML is not installed and could not be downloaded."); return 1; }
            var spans = pg.Detect(text);
            if (spans.Count == 0) Console.WriteLine("no PII entities found");
            foreach (var s in spans) Console.WriteLine($"[{s.Start},{s.End})  {s.Label,-32} {s.Confidence:P0}  '{s.Value}'");
            return 0;
        }

        default:
            Console.Error.WriteLine($"unknown command '{cmd}' - try: caveman-ml help");
            return 2;
    }
}
catch (Exception ex) when (ex is HttpRequestException or InvalidDataException or ArgumentException)
{
    Console.Error.WriteLine("error: " + ex.Message);
    return 1;
}
