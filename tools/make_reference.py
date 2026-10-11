"""Generates parity fixtures for CavemanML from the reference (Python) implementation in Synthelion 1.2.5.

Run with the Synthelion checkout on PYTHONPATH and torch installed:

    set PYTHONPATH=C:\\Sorgenti\\Personal\\Synthelion
    python tools/make_reference.py <models_dir> <out.json>

The C# tests load the JSON and assert that CavemanML reproduces the same per-word keep
probabilities, the same compressed text and the same PII spans.
"""
import json
import sys
from pathlib import Path

models = Path(sys.argv[1])
out = Path(sys.argv[2])

from synthelion.synthelionml import SynthelionMLCompressor  # noqa: E402
from synthelion.privacyguardml import PrivacyGuardMLDetector  # noqa: E402
from synthelion.core import _tokenize  # noqa: E402
from synthelion.detector import LanguageDetector  # noqa: E402
from synthelion.word_provider import FunctionWordProvider  # noqa: E402

COMPRESS = [
    "I would like to know if it is possible to receive information about cheap restaurants in Rome.",
    "Vorrei sapere se è possibile avere informazioni sui ristoranti economici a Roma entro domani.",
    "The quarterly financial report that we are going to send tomorrow morning to the whole board needs to be accurate.",
    "Please make sure that the server at 192.168.1.10:8080 is reachable before the deployment starts on 2026-10-11.",
    "Ich möchte gerne wissen, ob es möglich ist, Informationen über günstige Restaurants in Berlin zu bekommen.",
    "No, non possiamo accettare questa proposta perché non è sicura.",
    "Bonjour, je voudrais réserver une table pour quatre personnes demain soir au restaurant.",
]
PII = [
    "Call me on +39 333 1234567 or write to mario.rossi@example.com tomorrow.",
    "My IBAN is IT60X0542811101000000123456 and my card is 4539 1488 0343 6467.",
    "Il mio codice fiscale è RSSMRA80A01H501U, nato a Roma.",
    "Die Steuer-ID lautet 12345678901 und die Telefonnummer ist 030 1234567.",
    "Meeting at the office next Tuesday about the budget, nothing personal here.",
    "Posizione GPS 41.9028, 12.4964 per la consegna, targa AB123CD.",
]

provider = FunctionWordProvider()
det = LanguageDetector(provider)

ml = SynthelionMLCompressor(model_path=models / "synthelionml")
assert ml.is_available(), "synthelionml checkpoint not loadable"
cfg = ml._config

result = {"synthelionml": {"config": cfg, "cases": []}, "privacyguardml": {"cases": []}}
for text in COMPRESS:
    iso3 = det.detect(text)
    fw = provider.get_function_words(iso3)
    words = [t.text for t in _tokenize(text) if not t.is_punct]
    scores = ml._score_words(words, fw, iso3)
    result["synthelionml"]["cases"].append({
        "text": text, "iso3": iso3, "words": words, "scores": scores,
        "stop": [w.lower() in fw for w in words],
        "compressed": ml.compress(text, iso3),
        "compressed_budget_6": ml.compress(text, iso3, budget=6),
    })

pg = PrivacyGuardMLDetector(model_path=models / "privacyguardml", min_confidence=0.0)
assert pg.is_available(), "privacyguardml checkpoint not loadable"
import torch  # noqa: E402
from synthelion.privacyguardml import TAGS, word_features, N_FEATURES  # noqa: E402

for text in PII:
    toks = [t for t in _tokenize(text) if not t.is_punct]
    max_seq = pg._config.get("max_seq_len", 96)
    ids = [pg._vocab.encode_word(t.text) for t in toks[:max_seq]]
    feats = [word_features(t.text) for t in toks[:max_seq]]
    n = len(ids)
    ids_t = torch.tensor([ids + [0] * (max_seq - n)])
    f_t = torch.tensor([feats + [[0.0] * N_FEATURES] * (max_seq - n)], dtype=torch.float)
    m_t = torch.zeros((1, max_seq), dtype=torch.bool)
    m_t[0, :n] = True
    probs = pg._model.predict_tag_probs(ids_t, f_t, m_t)[0][:n]
    conf, tag = probs.max(dim=-1)
    spans = pg.detect(text)
    result["privacyguardml"]["cases"].append({
        "text": text, "words": [t.text for t in toks[:max_seq]],
        "tags": [TAGS[int(t)] for t in tag], "confidences": [float(c) for c in conf],
        "spans": [{"start": s.start, "end": s.end, "label": s.label, "confidence": s.confidence} for s in spans],
    })

out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
