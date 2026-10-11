<p align="center"><img src="https://raw.githubusercontent.com/francescopaolopassaro/caveman/main/caveman-icon.png" width="96" alt="Caveman"/></p>

# CavemanML

**The optional ML tier of the Caveman family - in pure C#.** Two small transformer models, trained on Wikipedia in 39 languages, running **offline on the CPU with no torch, no onnxruntime and no native code**:

| Model | What it does | Size | Source |
|---|---|---|---|
| **SynthelionML** | Learned per-token **prompt compression**: predicts which words can be dropped while the meaning survives, with a rank-based controller that hits a target ratio (70% by default) | 10 M params · 41 MB | [`digitalsolutionsai/Synthelion`](https://huggingface.co/digitalsolutionsai/Synthelion) |
| **PrivacyGuardML** | Learned **PII confirmation**: per-token BIO tagging over 16 PII categories, plugged into `Caveman.PrivacyGuard` as an opt-in second opinion | 6.5 M params · 26 MB | [`digitalsolutionsai/PrivacyGuardML`](https://huggingface.co/digitalsolutionsai/PrivacyGuardML) |

```
dotnet add package CavemanML
dotnet tool install -g CavemanML.Cli        # caveman-ml: install / verify / update / remove the models, run them from the shell
```

Port of the models shipped with Synthelion 1.2.5. The weights are read straight from the Hugging Face files (safetensors **and** the PyTorch `.bin` checkpoint), and the C# engine is verified against the PyTorch reference: keep-probabilities, BIO tags and PII spans agree to **< 1e-3**, compressed text and spans are identical.

> The rule-based core stays the default and needs none of this. CavemanML is for when you want *learned* behaviour: a compressor that generalises beyond the rules by looking at the surrounding words, and a PII detector that recognises a value by shape and context when no keyword is near.

## Models are downloaded and managed locally

Nothing to install by hand. The first time you ask for a model it is downloaded from Hugging Face into `~/.caveman/ml_models/<name>/`, **SHA-256 checked** against the Hub's digest, written atomically, and recorded in a manifest. After that everything is offline.

```csharp
var ml = await SynthelionMLCompressor.LoadAsync();            // downloads on first use, then loads from disk
var pg = await PrivacyGuardMLDetector.LoadAsync();

// or manage them explicitly
foreach (var s in ModelManager.GetStatus())                   // installed? where? how big? which revision?
    Console.WriteLine($"{s.Kind}: {(s.Installed ? s.Directory : "not installed")}");

await ModelManager.InstallAsync(MlModelKind.SynthelionML, progress: new Progress<(string File, long Done, long Total)>(p => ...));
var ok     = await ModelManager.VerifyAsync(MlModelKind.PrivacyGuardML);     // re-hash against the manifest
var update = await ModelManager.CheckUpdateAsync(MlModelKind.SynthelionML);  // newer revision on the Hub?
ModelManager.Remove(MlModelKind.SynthelionML);
```

| | |
|---|---|
| Where | `~/.caveman/ml_models/<name>/` - override with `CAVEMAN_ML_MODELS_DIR`, or per model with `CAVEMAN_ML_MODEL_SYNTHELIONML` / `CAVEMAN_ML_MODEL_PRIVACYGUARDML`; `~/.synthelion/ml_models/` (the Python port's folder) is found too |
| Integrity | every download is hashed and compared with the Hub's LFS SHA-256; a mismatch deletes the file; `verify` re-checks later and **detects a flipped byte** |
| Atomic | downloads go to `*.part` and are moved into place - a half-downloaded file is never mistaken for a model |
| Private mirror | `HF_TOKEN` is sent as a bearer token |
| Offline | `LoadAsync(download: false)` / `Load()` never touch the network |

### The `caveman-ml` tool

```
caveman-ml models                          status of every model
caveman-ml install [synthelionml|privacyguardml|all] [--force]
caveman-ml verify  [name|all]              re-hash installed files against the manifest
caveman-ml update  [name|all] [--install]  check (and fetch) a newer revision
caveman-ml remove  <name|all>
caveman-ml compress "text" [--lang ita] [--ratio 0.7] [--budget 12]
caveman-ml pii "text"                      PII entities found by PrivacyGuardML
```

## SynthelionML - learned prompt compression

```csharp
var ml = await SynthelionMLCompressor.LoadAsync();

string small = ml.Compress("I would like to know if it is possible to receive some information about cheap restaurants near the station.");
// "like know information about cheap station"

var r = ml.CompressResult(text, iso3: "ita", minCompression: 0.8);   // drop at least 80% of the words
var b = ml.Compress(text, budget: 12);                               // keep at most 12 words
```

Numbers, URLs, e-mails, IPs, dates, versions, **proper nouns and negations** are always kept verbatim; every other word is scored by the model; the lowest-scoring words are dropped until the target ratio is reached; a sentence is never left empty. Chinese and Japanese are segmented and joined without inflating the output.

On the three sample prompts it removes about **67-68% of the words in ~100 ms on one CPU thread** (model load 0.2 s). Honest comparison: it lands in the same range as Caveman's `Aggressive` level - what you gain is *context-sensitivity* (the same word is kept or dropped depending on its neighbours), not a different ratio; check it on your own prompts before swapping.

## PrivacyGuardML - learned PII confirmation

```csharp
using var analyzer = new PrivacyAnalyzer { MlDetector = await PrivacyGuardMLDetector.LoadAsync() };
var result = analyzer.Analyze("ssn 123-45-6789 lo trovi nel file");
// result.MlAssistedCount == 1: a bare value the context gate would have rejected, confirmed by the model
```

`PrivacyAnalyzer`'s core is regex + checksum (+ a context keyword for rules with no checksum), and stays zero-ML. The model is only an **opt-in confirmation signal**: a value that matches a `requires_context` rule but has no keyword nearby is confirmed by an overlapping span of the *right label* (a "phone number" span can never confirm an ID rule). **A failed checksum still vetoes the detection** whatever the model says - it can recover recall the strict gate trades away, never add a false positive the validator would reject.

Use it directly for spans: `pg.Detect(text)` returns `MlSpan(Start, End, Value, Label, Confidence)`; `pg.Tag(words)` returns the raw BIO tags.

Categories: phone, e-mail, credit card, IBAN, national ID, tax ID, SSN, GPS, vehicle plate, badge ID, business ID, secret, social handle, legal case, booking code, minor age.

## How it works

`word embedding (+ char-n-gram hash for out-of-vocabulary words) + feature projection + positional embedding -> 2 x transformer encoder layer (d=128, 4 heads, GELU) -> linear head`. The engine implements exactly `torch.nn.TransformerEncoderLayer` in eval mode in plain C# (vectorised dot products); the PyTorch pickle is read by a small built-in reader, so there is no dependency beyond `Caveman` and `Caveman.PrivacyGuard`.

## Limits

* The models are small (6-10 M parameters) and trained by self-distillation: SynthelionML imitates Caveman's rule-based compressor, PrivacyGuardML finds synthetic PII it was trained to find. Neither is a general-purpose NER or summariser.
* *Is-stop-word* is one of SynthelionML's inputs and Caveman and Synthelion curate their lists independently, so scores on the same text differ slightly between the two ports (mean difference < 0.1, engine itself exact).
* PrivacyGuardML tags by shape and context; on a sentence with no PII-like token it returns nothing, and low-confidence spans (< 0.6 by default, `MinConfidence`) are dropped.

## Requirements

.NET 8 - depends on [`Caveman`](https://www.nuget.org/packages/Caveman) and [`Caveman.PrivacyGuard`](https://www.nuget.org/packages/Caveman.PrivacyGuard) (>= 1.4.0).

## License

MIT - see `LICENSE`. Models: MIT, (c) Francesco Paolo Passaro - Digitalsolutions.it.
