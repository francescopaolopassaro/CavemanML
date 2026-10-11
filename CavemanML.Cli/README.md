<p align="center"><img src="https://raw.githubusercontent.com/francescopaolopassaro/caveman/main/caveman-icon.png" width="96" alt="Caveman"/></p>

# caveman-ml

Command-line manager and runner for the [CavemanML](https://www.nuget.org/packages/CavemanML) models - **SynthelionML** (learned prompt compression) and **PrivacyGuardML** (learned PII detection), both from Hugging Face ([`digitalsolutionsai`](https://huggingface.co/digitalsolutionsai)).

```
dotnet tool install -g CavemanML.Cli
```

```
caveman-ml models                          status of every model (installed? where? how big? which revision?)
caveman-ml install [synthelionml|privacyguardml|all] [--force]
caveman-ml verify  [name|all]              re-hash installed files against the manifest (SHA-256)
caveman-ml update  [name|all] [--install]  check - and optionally fetch - a newer revision
caveman-ml remove  <name|all>
caveman-ml compress "text" [--lang ita] [--ratio 0.7] [--budget 12]
caveman-ml pii "text"                      PII entities found by PrivacyGuardML
```

Models are downloaded into `~/.caveman/ml_models/<name>/` (override with `CAVEMAN_ML_MODELS_DIR`), SHA-256 checked, written atomically and recorded in a manifest. After the download everything runs offline on the CPU. `HF_TOKEN` is honoured for private mirrors.

```
$ caveman-ml compress "I would like to know if it is possible to receive some information about cheap restaurants near the station."
like know information about cheap station
[19 -> 6 words, 68% saved]

$ caveman-ml pii "Call +39 333 1234567 or mail mario.rossi@example.com"
[5,20)  phone number         95%  '+39 333 1234567'
[29,52) email address       100%  'mario.rossi@example.com'
```

MIT - (c) 2026 Francesco Paolo Passaro - Digitalsolutions.it
