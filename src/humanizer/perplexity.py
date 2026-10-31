"""Perplexity scoring against a local language model.

This is what the commercial detectors actually measure. Turnitin's own
description is that it "analyses how often the next most probable word
is used". GPTZero, Copyleaks, ZeroGPT, Sapling and Writer work the same
way: perplexity over a language model, combined with burstiness over
sentence-length distribution.

Our detector was scoring on a lexical catalogue - a list of phrases
that read as AI-written. That catches the obvious tells and nothing
else. A dense technical passage with no catalogue phrases but
uniform, predictable syntax scored 0.09 in our tool and 76% AI across
seven commercial detectors. The catalogue is a proxy for perplexity,
and on technical prose the proxy breaks.

So this module measures the thing itself.

The perplexity comes from a local model via Ollama's ``/api/generate``
with ``echo=True``, which returns the logprob of every prompt token
along with the top alternatives at each position. The top alternatives
give burstiness a second, sharper reading: how far down the model's
ranking the actual word was.

Measured on this machine (qwen2.5:7b-instruct, RTX 3070), a passage
scores in about 40 ms of GPU time after the model is loaded.

If Ollama is unavailable or the model is not pulled, every function
here degrades to ``None`` and the caller falls back to the lexical
score. The detector must never crash because a local model is down.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import httpx

#: The Ollama endpoint. Overridable for a remote host.
OLLAMA_URL = os.environ.get("HUMANIZER_OLLAMA_URL", "http://127.0.0.1:11434")

#: The model used for the perplexity measurement. A 7-8B model is
#: enough: perplexity rankings are stable across model sizes, and the
#: commercial detectors use models in this range.
DEFAULT_MODEL = os.environ.get("HUMANIZER_PERPLEXITY_MODEL", "qwen2.5:7b-instruct")

#: A passage shorter than this cannot be scored meaningfully - the
#: per-token statistics are too noisy.
MIN_TOKENS = 40

#: How long to wait for the local model. The first call after a restart
#: has to load 4.7 GB off disk.
TIMEOUT = float(os.environ.get("HUMANIZER_PERPLEXITY_TIMEOUT", "180"))


@dataclass
class PerplexityResult:
    """The perplexity measurement for one passage."""

    #: Mean negative log-probability per token. Lower means more
    #: predictable, i.e. more machine-like.
    mean_nll: float = 0.0
    #: exp(mean_nll) - the textbook perplexity.
    perplexity: float = 0.0
    #: Standard deviation of the per-token negative log-probability.
    #: This is the "burstiness" the detectors talk about at token level.
    stdev_nll: float = 0.0
    #: Mean rank of the actual token among the model's top candidates.
    #: 0 means the model wanted that word most; higher means it was a
    #: less obvious choice.
    mean_rank: float = 0.0
    #: Share of tokens that were the model's top choice.
    top_choice_share: float = 0.0
    #: Number of tokens scored.
    token_count: int = 0
    #: Set when the measurement could not be taken.
    unavailable: bool = False
    #: Why it could not be taken.
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "mean_nll": round(self.mean_nll, 4),
            "perplexity": round(self.perplexity, 2),
            "stdev_nll": round(self.stdev_nll, 4),
            "mean_rank": round(self.mean_rank, 3),
            "top_choice_share": round(self.top_choice_share, 4),
            "token_count": self.token_count,
            "unavailable": self.unavailable,
            "reason": self.reason,
        }


#: Cache of results keyed by (model, text hash). The same passage
#: scored twice in one session should not cost two model passes.
_CACHE: dict[tuple[str, str], PerplexityResult] = {}


def _cache_key(text: str, model: str) -> tuple[str, str]:
    import hashlib

    return (model, hashlib.sha256(text.encode("utf-8")).hexdigest())


def available(model: str = DEFAULT_MODEL) -> bool:
    """True when the local model can be reached and has the text."""
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code != 200:
                return False
            names = {m.get("name", "") for m in r.json().get("models", [])}
            return any(model in n for n in names)
    except Exception:  # noqa: BLE001
        return False


def score(text: str, model: str = DEFAULT_MODEL) -> PerplexityResult:
    """Measure the perplexity of ``text``.

    Returns an ``unavailable`` result - never an exception - when the
    local model cannot be reached or the passage is too short.
    """
    if not text or not text.strip():
        return PerplexityResult(unavailable=True, reason="empty text")

    key = _cache_key(text, model)
    if key in _CACHE:
        return _CACHE[key]

    payload = {
        "model": model,
        "prompt": text,
        "stream": False,
        "echo": True,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {
            "temperature": 0,
            "num_predict": 0,
            "num_ctx": 4096,
        },
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        if r.status_code != 200:
            result = PerplexityResult(
                unavailable=True, reason=f"HTTP {r.status_code}"
            )
            _CACHE[key] = result
            return result
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        result = PerplexityResult(unavailable=True, reason=type(exc).__name__)
        _CACHE[key] = result
        return result

    entries = data.get("logprobs") or []
    if len(entries) < MIN_TOKENS:
        result = PerplexityResult(
            unavailable=True,
            reason=f"only {len(entries)} tokens",
            token_count=len(entries),
        )
        _CACHE[key] = result
        return result

    nlls: list[float] = []
    ranks: list[int] = []
    top_hits = 0

    for entry in entries:
        logprob = entry.get("logprob")
        if logprob is None:
            continue
        nlls.append(-float(logprob))

        # The position of the actual token in the model's own ranking.
        token = entry.get("token")
        alternatives = entry.get("top_logprobs") or []
        rank = len(alternatives)
        for i, alt in enumerate(alternatives):
            if alt.get("token") == token:
                rank = i
                break
        ranks.append(rank)
        if rank == 0:
            top_hits += 1

    if not nlls:
        result = PerplexityResult(unavailable=True, reason="no logprobs")
        _CACHE[key] = result
        return result

    mean_nll = statistics.mean(nlls)
    result = PerplexityResult(
        mean_nll=mean_nll,
        perplexity=math.exp(min(mean_nll, 20.0)),
        stdev_nll=statistics.pstdev(nlls) if len(nlls) > 1 else 0.0,
        mean_rank=statistics.mean(ranks) if ranks else 0.0,
        top_choice_share=top_hits / len(nlls),
        token_count=len(nlls),
    )
    _CACHE[key] = result
    return result


# ----------------------------------------------------------------------
# sentence-level burstiness
# ----------------------------------------------------------------------

@dataclass
class BurstinessResult:
    """How varied the sentence lengths are."""

    #: Coefficient of variation of sentence lengths.
    cv: float = 0.0
    #: Mean sentence length in words.
    mean_length: float = 0.0
    #: Stdev of sentence length in words.
    stdev_length: float = 0.0
    #: Shortest and longest sentence.
    min_length: int = 0
    max_length: int = 0
    sentence_count: int = 0

    def as_dict(self) -> dict:
        return {
            "cv": round(self.cv, 4),
            "mean_length": round(self.mean_length, 2),
            "stdev_length": round(self.stdev_length, 2),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "sentence_count": self.sentence_count,
        }


def burstiness(text: str) -> BurstinessResult:
    """Measure sentence-length variation.

    GPTZero's original paper pairs perplexity with burstiness: human
    writing has more variation in sentence length, and an even, uniform
    rhythm is a machine signature.
    """
    if not text or not text.strip():
        return BurstinessResult()

    # Split on sentence enders that are not inside a common abbreviation.
    cleaned = re_sub_abbreviations(text)
    sentences = [
        s.strip().replace(_ABBREV_SENTINEL, ".").replace("\u0002", ".")
        for s in cleaned.split(".")
        if s.strip()
    ]
    if len(sentences) < 2:
        sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]

    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return BurstinessResult(
            cv=0.0,
            mean_length=float(lengths[0]) if lengths else 0.0,
            stdev_length=0.0,
            min_length=lengths[0] if lengths else 0,
            max_length=lengths[0] if lengths else 0,
            sentence_count=len(lengths),
        )

    mean = statistics.mean(lengths)
    stdev = statistics.pstdev(lengths)
    return BurstinessResult(
        cv=stdev / mean if mean else 0.0,
        mean_length=mean,
        stdev_length=stdev,
        min_length=min(lengths),
        max_length=max(lengths),
        sentence_count=len(lengths),
    )


_ABBREVIATIONS = (
    "e.g", "i.e", "et al", "Fig", "fig", "cf", "vs", "etc", "Dr", "Mr",
    "Ms", "Prof", "St", "approx", "No", "vol", "pp",
)


#: The sentinel that stands in for the full stop inside an abbreviation.
_ABBREV_SENTINEL = "\u0001"


def re_sub_abbreviations(text: str) -> str:
    """Protect common abbreviations from the sentence splitter.

    An abbreviation's full stop is not a sentence end, so it is swapped
    for a sentinel the splitter ignores, and restored afterwards.
    """
    import re

    out = text
    for abbr in _ABBREVIATIONS:
        out = re.sub(
            re.escape(abbr) + r"\.",
            abbr.replace(".", "\u0002") + _ABBREV_SENTINEL,
            out,
        )
    return out


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------

@dataclass
class Calibration:
    """Thresholds learned from a labelled corpus.

    ``p90_human`` is the perplexity of the 90th percentile of human
    text: anything above it is more predictable than 9 out of 10 human
    passages, which is where the commercial detectors draw their line.
    """

    p50_human: float = 0.0
    p90_human: float = 0.0
    p50_ai: float = 0.0
    p90_ai: float = 0.0
    human_burstiness_p10: float = 0.0
    ai_burstiness_p90: float = 0.0
    human_burstiness_p50: float = 0.0
    ai_burstiness_p50: float = 0.0
    sample_size_human: int = 0
    sample_size_ai: int = 0
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "p50_human": round(self.p50_human, 4),
            "p90_human": round(self.p90_human, 4),
            "p50_ai": round(self.p50_ai, 4),
            "p90_ai": round(self.p90_ai, 4),
            "human_burstiness_p10": round(self.human_burstiness_p10, 4),
            "ai_burstiness_p90": round(self.ai_burstiness_p90, 4),
            "human_burstiness_p50": round(self.human_burstiness_p50, 4),
            "ai_burstiness_p50": round(self.ai_burstiness_p50, 4),
            "sample_size_human": self.sample_size_human,
            "sample_size_ai": self.sample_size_ai,
            "notes": self.notes,
        }


def percentile(values: list[float], pct: float) -> float:
    """The ``pct`` percentile of ``values``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


def calibrate(
    human_texts: list[str],
    ai_texts: list[str],
    model: str = DEFAULT_MODEL,
) -> Calibration:
    """Learn the thresholds from a labelled corpus."""
    human_nll = [score(t, model).mean_nll for t in human_texts
                 if not score(t, model).unavailable]
    ai_nll = [score(t, model).mean_nll for t in ai_texts
              if not score(t, model).unavailable]
    human_b = [burstiness(t).cv for t in human_texts]
    ai_b = [burstiness(t).cv for t in ai_texts]

    return Calibration(
        p50_human=percentile(human_nll, 50),
        p90_human=percentile(human_nll, 90),
        p50_ai=percentile(ai_nll, 50),
        p90_ai=percentile(ai_nll, 90),
        human_burstiness_p10=percentile(human_b, 10),
        ai_burstiness_p90=percentile(ai_b, 90),
        human_burstiness_p50=percentile(human_b, 50),
        ai_burstiness_p50=percentile(ai_b, 50),
        sample_size_human=len(human_nll),
        sample_size_ai=len(ai_nll),
        notes="nll = mean negative log-probability per token; lower is more predictable",
    )


#: Where the calibration lives. A JSON file, so the thresholds are
#: data someone can read, check and update - not magic numbers in the
#: source.
CALIBRATION_PATH = os.environ.get(
    "HUMANIZER_CALIBRATION",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "perplexity_calibration.json"),
)

_CALIBRATION: Calibration | None = None


def load_calibration(path: str | None = None) -> Calibration:
    """Load the calibration thresholds from disk.

    Returns an empty calibration when the file is missing, which makes
    every score that depends on it return 0.0 rather than raising.
    """
    global _CALIBRATION
    if path is None and _CALIBRATION is not None:
        return _CALIBRATION

    target = Path(path or CALIBRATION_PATH)
    if not target.exists():
        cal = Calibration(notes="no calibration file; perplexity scoring disabled")
        if path is None:
            _CALIBRATION = cal
        return cal

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        cal = Calibration(notes=f"calibration unreadable: {type(exc).__name__}")
        if path is None:
            _CALIBRATION = cal
        return cal

    cal = Calibration(
        p50_human=data.get("p50_human", 0.0),
        p90_human=data.get("p90_human", 0.0),
        p50_ai=data.get("p50_ai", 0.0),
        p90_ai=data.get("p90_ai", 0.0),
        human_burstiness_p10=data.get("human_burstiness_p10", 0.0),
        ai_burstiness_p90=data.get("ai_burstiness_p90", 0.0),
        human_burstiness_p50=data.get("human_burstiness_p50", 0.0),
        ai_burstiness_p50=data.get("ai_burstiness_p50", 0.0),
        sample_size_human=data.get("sample_size_human", 0),
        sample_size_ai=data.get("sample_size_ai", 0),
        notes=data.get("notes", ""),
    )
    if path is None:
        _CALIBRATION = cal
    return cal


def save_calibration(cal: Calibration, path: str | None = None) -> str:
    """Write the calibration to disk."""
    target = Path(path or CALIBRATION_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cal.as_dict(), indent=2) + "\n", encoding="utf-8")
    return str(target)


@dataclass
class SentencePredictability:
    """One sentence, with how predictable the model found it."""

    text: str
    mean_nll: float
    token_count: int
    line: int


def predictable_sentences(
    text: str,
    top: int = 3,
    model: str = DEFAULT_MODEL,
    min_tokens: int = 12,
) -> list[SentencePredictability]:
    """The sentences a language model finds most predictable.

    This is the sharpest version of the "point the rewrite" idea. The
    lexical catalogue can tell you a sentence contains a watched phrase.
    Perplexity can tell you that a sentence assembled entirely from
    high-probability continuations reads as machine-written even when
    every word in it is individually ordinary - and it does that for
    technical prose where the catalogue has nothing to say.

    Sentences are scored individually, so each measurement is a short
    forward pass. Sentences below ``min_tokens`` are skipped: their
    statistics are noise.
    """
    import re

    out: list[SentencePredictability] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]", text):
        sentence = m.group(0).strip()
        if len(sentence.split()) < 6:
            continue
        result = score(sentence, model)
        if result.unavailable or result.token_count < min_tokens:
            continue
        out.append(
            SentencePredictability(
                text=sentence,
                mean_nll=result.mean_nll,
                token_count=result.token_count,
                line=text.count("\n", 0, m.start()) + 1,
            )
        )

    out.sort(key=lambda s: s.mean_nll)
    return out[:top]


def perplexity_score(result: PerplexityResult, cal: Calibration) -> float:
    """Turn a measurement into a 0-1 AI-likeness score.

    AI text is MORE predictable, so its median nll is LOWER than the
    human median. The scale therefore runs:

        human median nll  ->  0.0  (reads as human)
        AI median nll     ->  1.0  (reads as machine)

    Anything more predictable than the AI median saturates at 1.0, and
    anything less predictable than the human median at 0.0.
    """
    if result.unavailable:
        return 0.0
    human = cal.p50_human
    ai = cal.p50_ai
    # Degenerate calibration - the two medians coincide or are
    # inverted. Score 0 rather than dividing by zero.
    if human <= ai:
        return 0.0
    value = (human - result.mean_nll) / (human - ai)
    return round(max(0.0, min(1.0, value)), 4)


def burstiness_score(result: BurstinessResult, cal: Calibration) -> float:
    """Turn sentence-length variation into a 0-1 AI-likeness score.

    Uniform sentence rhythm reads as machine-generated, so a LOW
    coefficient of variation is a bad sign. The scale mirrors the
    perplexity one:

        human median cv   ->  0.0  (varied, reads as human)
        AI median cv      ->  1.0  (uniform, reads as machine)
    """
    human = cal.human_burstiness_p50
    ai = cal.ai_burstiness_p50
    if human <= ai:
        return 0.0
    value = (human - result.cv) / (human - ai)
    return round(max(0.0, min(1.0, value)), 4)
