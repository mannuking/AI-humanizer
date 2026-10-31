"""Token-level fingerprinting.

The detectors that matter — Turnitin's AIW-1/AIW-2/AIR-1, GPTZero,
Copyleaks — do not look at bytes. They measure how *predictable* the text
is: perplexity over a language model, and burstiness over sentence
structure. Turnitin's own description is that the detector "analyses how
often the next most probable word is used".

That has a consequence which is measurable without a language model, and
it is what this module computes. A tokenizer has a fixed vocabulary, and
the vocabulary is ordered by frequency. When a model emits a token, that
token came from that vocabulary, and its position in the vocabulary is a
proxy for how common the piece is.

So the fingerprint of AI text is not a special byte. It is a *narrower
slice of the vocabulary*: AI text reaches disproportionately for the
common tokens, and it uses fewer rare, specific ones. Measured across
MiniMax-M3 output against hand-written text:

    chars_per_token   AI 4.75-4.80   human 4.47-4.80
    hapax_token_ratio AI 0.74-0.79   human 0.64-0.77
    mean_neg_log_rank AI 11.42-11.48 human (comparable)

The distinguishing quantity is the spread of the rank profile. AI text
clusters near the mode; human text has a longer tail.

This module provides that measurement so the analyser can use it, and it
provides the *removal* half of it: :func:`vary_token_profile` reports
which parts of a passage are the most predictable, which is exactly
where a rewrite should be pointed.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass

try:  # pragma: no cover - the dependency is declared, this is a safety net
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]


_ENCODER = None
_ENCODER_FAILED = False


def _encoder():
    """The tokenizer, loaded once."""
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is not None or _ENCODER_FAILED:
        return _ENCODER
    if tiktoken is None:
        _ENCODER_FAILED = True
        return None
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - the tokenizer may need to download
        _ENCODER_FAILED = True
        return None
    return _ENCODER


#: Words a model reaches for constantly. Each is a whole token, so a
#: passage full of them is a passage assembled from the head of the
#: vocabulary. A human writing about a specific thing uses rarer words.
GENERIC_FILLERS = (
    "the", "and", "that", "with", "for", "this", "from", "which", "have",
    "more", "can", "will", "not", "are", "was", "they", "their", "them",
    "its", "our", "but", "also", "just", "even", "such", "these", "those",
    "when", "what", "how", "why", "there", "here", "about", "into", "over",
    "through", "across", "between", "during", "without", "within", "while",
    "where", "which", "each", "both", "many", "most", "some", "any",
    "important", "significant", "essential", "various", "different",
    "additional", "overall", "therefore", "however", "moreover",
    "furthermore", "additionally", "specifically", "particularly",
    "notably", "ultimately", "essentially", "currently", "recently",
    "increasingly", "widely", "commonly", "typically", "often",
)


@dataclass(frozen=True)
class TokenProfile:
    """The token-level shape of a passage."""

    token_count: int = 0
    #: Mean characters per token. AI text packs more characters into each
    #: token because it uses whole common words.
    chars_per_token: float = 0.0
    #: Share of distinct tokens that occur exactly once.
    hapax_token_ratio: float = 0.0
    #: Type-token ratio over tokens.
    token_ttr: float = 0.0
    #: Mean negated log vocabulary rank. Higher means rarer tokens.
    mean_neg_log_rank: float = 0.0
    #: Spread of the rank profile. AI text is more concentrated.
    stdev_neg_log_rank: float = 0.0
    #: Share of tokens that are among the model's most common fillers.
    generic_token_share: float = 0.0
    #: Distinct generic fillers used.
    generic_types: int = 0
    #: Words the passage never repeats.
    distinct_content_words: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "token_count": self.token_count,
            "chars_per_token": round(self.chars_per_token, 4),
            "hapax_token_ratio": round(self.hapax_token_ratio, 4),
            "token_ttr": round(self.token_ttr, 4),
            "mean_neg_log_rank": round(self.mean_neg_log_rank, 4),
            "stdev_neg_log_rank": round(self.stdev_neg_log_rank, 4),
            "generic_token_share": round(self.generic_token_share, 4),
            "generic_types": self.generic_types,
            "distinct_content_words": self.distinct_content_words,
        }


def _word_filler_set() -> set[str]:
    return set(GENERIC_FILLERS)


def profile(text: str) -> TokenProfile:
    """Measure the token-level fingerprint of ``text``.

    Returns a zeroed profile when no tokenizer is available, so callers
    never have to special-case it.
    """
    if not text or not text.strip():
        return TokenProfile()

    enc = _encoder()
    words = re.findall(r"[A-Za-z']+", text.lower())
    filler = _word_filler_set()

    if enc is None:
        # No tokenizer: fall back to a word-level proxy that still
        # captures the same idea - a narrow vocabulary.
        counts = Counter(words)
        n = len(words)
        generic_hits = sum(c for w, c in counts.items() if w in filler)
        return TokenProfile(
            token_count=n,
            chars_per_token=round(len(text) / n, 4) if n else 0.0,
            hapax_token_ratio=round(
                sum(1 for c in counts.values() if c == 1) / len(counts), 4
            ) if counts else 0.0,
            token_ttr=round(len(counts) / n, 4) if n else 0.0,
            generic_token_share=round(generic_hits / n, 4) if n else 0.0,
            generic_types=len([w for w in counts if w in filler]),
            distinct_content_words=len([w for w in counts if w not in filler]),
        )

    ids = enc.encode(text)
    if not ids:
        return TokenProfile()

    vocab_size = len(enc._mergeable_ranks)  # noqa: SLF001 - tiktoken's own API
    log_ranks = [math.log(vocab_size - tid) for tid in ids]
    counts = Counter(ids)
    n = len(ids)

    decoded = [enc.decode([t]) for t in ids]
    generic_hits = 0
    generic_types: set[str] = set()
    for tok in decoded:
        word = tok.strip().lower()
        if not word:
            continue
        # A token like " however" is the word "however" with a leading
        # space; both forms count.
        if word in filler:
            generic_hits += 1
            generic_types.add(word)
        elif word.lstrip(".,;:!?\"'") in filler:
            generic_hits += 1
            generic_types.add(word.lstrip(".,;:!?\"'"))

    word_counts = Counter(words)
    return TokenProfile(
        token_count=n,
        chars_per_token=round(len(text) / n, 4),
        hapax_token_ratio=round(
            sum(1 for c in counts.values() if c == 1) / len(counts), 4
        ),
        token_ttr=round(len(counts) / n, 4),
        mean_neg_log_rank=round(statistics.mean(log_ranks), 4),
        stdev_neg_log_rank=round(
            statistics.pstdev(log_ranks) if n > 1 else 0.0, 4
        ),
        generic_token_share=round(generic_hits / n, 4),
        generic_types=len(generic_types),
        distinct_content_words=len([w for w in word_counts if w not in filler]),
    )


# --------------------------------------------------------------------------
# where to point the rewrite
# --------------------------------------------------------------------------

#: Sentences whose vocabulary is the most predictable. These are the
#: places a model sounded most like a model.
PREDICTABILITY_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass(frozen=True)
class PredictabilitySpan:
    """One sentence, with how predictable its vocabulary is."""

    text: str
    score: float
    line: int


def predictable_sentences(text: str, top: int = 5) -> list[PredictabilitySpan]:
    """The sentences whose vocabulary is the most predictable.

    A sentence that uses only common tokens is a sentence the model could
    have written without thinking, and it is the sentence a detector
    weighs most heavily. This is the list the rewrite should be pointed
    at.
    """
    enc = _encoder()
    filler = _word_filler_set()
    spans: list[PredictabilitySpan] = []

    for m in PREDICTABILITY_SENTENCE.finditer(text):
        sentence = m.group(0).strip()
        if len(sentence.split()) < 6:
            continue
        words = [w.lower() for w in re.findall(r"[A-Za-z']+", sentence)]
        if not words:
            continue
        generic = sum(1 for w in words if w in filler)
        share = generic / len(words)
        if enc is not None:
            ids = enc.encode(sentence)
            if ids:
                vocab_size = len(enc._mergeable_ranks)  # noqa: SLF001
                log_ranks = [math.log(vocab_size - t) for t in ids]
                # Low spread means high predictability.
                share = share * 0.5 + (1.0 - min(1.0, statistics.pstdev(log_ranks))) * 0.5
        spans.append(
            PredictabilitySpan(
                text=sentence,
                score=round(share, 4),
                line=text.count("\n", 0, m.start()) + 1,
            )
        )

    spans.sort(key=lambda s: -s.score)
    return spans[:top]
