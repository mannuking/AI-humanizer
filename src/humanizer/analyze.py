"""Detection engine: find the AI-writing signature in a passage.

Two layers:

* **Lexical** - the named detectors in :mod:`humanizer.data`, which catch
  stock phrases, inflated significance, chatbot artifacts and formatting
  tells.
* **Statistical** - the shape of the prose: sentence-length variance
  (burstiness), type-token ratio, punctuation entropy and readability.
  Human writing varies; LLM output converges.

The scores are a *style* estimate, not proof of authorship. Nothing here
can establish who wrote a text, and the tool should not claim that it can.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from . import data
from .document import DocumentFinding, detect_document_patterns
from .guards import filter_findings
from .thresholds import CountFinding, detect_counted_patterns

# --------------------------------------------------------------------------
# lexical layer
# --------------------------------------------------------------------------

#: Maps a detector's ``shared_consts`` entry to the compiled list.
_SHARED_LOOKUP = {
    "SIGNIFICANCE_PHRASES": data.SIGNIFICANCE_PHRASES,
    "PROMOTIONAL_WORDS": data.PROMOTIONAL_WORDS,
    "VAGUE_ATTRIBUTION_PHRASES": data.VAGUE_ATTRIBUTION_PHRASES,
    "CHALLENGES_PHRASES": data.CHALLENGES_PHRASES,
    "COPULA_AVOIDANCE": data.COPULA_AVOIDANCE,
    "PRODUCT_POSITIONING_FOG": data.PRODUCT_POSITIONING_FOG,
    "HIDDEN_UNICODE_CHARS": ("(?:\\u200B|\\u200C|\\u200D|\\u2060|\\uFEFF|\\u00AD)",),
    "NON_BREAKING_SPACES": ("(?:\\u00A0|\\u202F)",),
}


@dataclass(frozen=True)
class Finding:
    """One detected pattern occurrence."""

    detector_id: int
    detector: str
    category: str
    match: str
    line: int
    column: int
    suggestion: str
    weight: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "detector": self.detector,
            "category": self.category,
            "match": self.match,
            "line": self.line,
            "column": self.column,
            "suggestion": self.suggestion,
            "weight": self.weight,
        }


#: Detectors whose logic is a threshold or a condition rather than a
#: plain regex match. Their extracted regexes are only unit extractors -
#: running them as plain scans makes "/[.!?]+/" match ordinary punctuation
#: and "/^[A-Z]/" match any capital letter. The real logic is in
#: :mod:`humanizer.thresholds`.
_THRESHOLD_DETECTORS = {
    "Synonym cycling",
    "Title Case headings",
    "Inline-header lists",
    "Excessive structure",
    "Em dash overuse",
    "Boldface overuse",
    "Emoji overuse",
}


def _compile_all() -> list[tuple[dict, list[re.Pattern]]]:
    """Pre-compile every detector's regexes once, at import time."""
    compiled: list[tuple[dict, list[re.Pattern]]] = []
    for det in data.PATTERNS:
        if det["name"] in _THRESHOLD_DETECTORS:
            continue
        patterns: list[re.Pattern] = []
        for rx in det["regexes"]:
            try:
                patterns.append(re.compile(rx, re.IGNORECASE | re.MULTILINE))
            except re.error:
                continue
        for const in det.get("shared_consts", ()):
            for rx in _SHARED_LOOKUP.get(const, ()):
                try:
                    patterns.append(re.compile(rx, re.IGNORECASE | re.MULTILINE))
                except re.error:
                    continue
        if patterns:
            compiled.append((det, patterns))
    return compiled


_COMPILED = _compile_all()

#: Phrase patterns, kept separately because several detectors draw on them.
_PHRASE_PATTERNS = [
    (rx, tier, fix)
    for rx, tier, fix in data.AI_PHRASES
]


def _line_col(text: str, index: int) -> tuple[int, int]:
    """Translate a character offset into a 1-based (line, column) pair."""
    line = text.count("\n", 0, index) + 1
    last_nl = text.rfind("\n", 0, index)
    column = index - last_nl if last_nl >= 0 else index + 1
    return line, column


def _suggest_for(detector_name: str) -> str:
    hints = {
        "Significance inflation": "Replace the inflated claim with a concrete fact.",
        "Notability name-dropping": "Cite one specific claim from one source.",
        "Superficial -ing analyses": "Give the point its own sentence with specifics.",
        "Promotional language": "Replace the marketing register with plain description.",
        "Vague attributions": "Name a specific source, or drop the attribution.",
        "Formulaic challenges": "State the actual difficulty instead of the formula.",
        "AI vocabulary": "Swap the flagged word for a plain one.",
        "Copula avoidance": "Use a plain verb instead of avoiding 'is'/'are'.",
        "Negative parallelisms": "State the positive claim directly.",
        "Rule of three": "Break the three-part list or make it two.",
        "Synonym cycling": "Keep one word instead of rotating synonyms.",
        "False ranges": "Give the real range rather than 'from X to Y'.",
        "Em dash overuse": "Replace most em dashes with commas or periods.",
        "Boldface overuse": "Remove the mechanical bold emphasis.",
        "Inline-header lists": "Turn the pseudo-header list into prose.",
        "Title Case headings": "Use sentence case for the heading.",
        "Emoji overuse": "Remove emoji decoration from professional text.",
        "Curly quotes": "Use straight quotes.",
        "Chatbot artifacts": "Delete the leftover chatbot phrasing.",
        "Cutoff disclaimers": "Delete the knowledge-cutoff disclaimer.",
        "Sycophantic tone": "Answer directly instead of opening with praise.",
        "Filler phrases": "Delete the filler and start with the content.",
        "Excessive hedging": "State the claim without the hedge stack.",
        "Generic conclusions": "End with something specific, not a summary.",
        "Reasoning chain artifacts": "Delete the visible reasoning scaffold.",
        "Excessive structure": "Flatten the over-structured formatting.",
        "Confidence calibration": "State the confidence once, not repeatedly.",
        "Acknowledgment loops": "Delete the acknowledgement loop.",
        "Invisible unicode obfuscation": "Remove the hidden characters.",
        "Product positioning fog": "Replace the vague positioning with specifics.",
    }
    return hints.get(detector_name, "Review this phrase.")


def detect_patterns(text: str) -> list[Finding]:
    """Run every named detector over ``text``.

    Returns the findings, heaviest first. A pattern is reported once per
    detector regardless of how many times it occurs, so a text that repeats
    one phrase does not drown out every other signal.
    """
    findings: list[Finding] = []
    if not text or not text.strip():
        return findings

    for det, patterns in _COMPILED:
        seen: set[str] = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                matched = m.group(0).strip()
                if not matched or matched.lower() in seen:
                    continue
                seen.add(matched.lower())
                line, column = _line_col(text, m.start())
                findings.append(
                    Finding(
                        detector_id=det["id"],
                        detector=det["name"],
                        category=det["category"],
                        match=matched,
                        line=line,
                        column=column,
                        suggestion=_suggest_for(det["name"]),
                        weight=det["weight"],
                    )
                )

    # Stock phrases, reported against the vocabulary detector.
    for rx, tier, fix in _PHRASE_PATTERNS:
        try:
            pattern = re.compile(rx, re.IGNORECASE)
        except re.error:
            continue
        for m in pattern.finditer(text):
            matched = m.group(0).strip()
            if not matched:
                continue
            line, column = _line_col(text, m.start())
            findings.append(
                Finding(
                    detector_id=7,
                    detector="AI vocabulary",
                    category="language",
                    match=matched,
                    line=line,
                    column=column,
                    suggestion=fix if fix != "(remove)" else "Remove this phrase.",
                    weight=5 if tier == 1 else (3 if tier == 2 else 2),
                )
            )

    # Tiered vocabulary words.
    for tier_no, words in ((1, data.TIER_1), (2, data.TIER_2), (3, data.TIER_3)):
        weight = {1: 4, 2: 2, 3: 1}[tier_no]
        lowered = text.lower()
        for word in words:
            if not word:
                continue
            # Word-boundary match for single words; substring for phrases.
            if " " in word or "-" in word:
                if word.lower() in lowered:
                    idx = lowered.index(word.lower())
                    line, column = _line_col(text, idx)
                    findings.append(
                        Finding(
                            detector_id=7,
                            detector="AI vocabulary",
                            category="language",
                            match=word,
                            line=line,
                            column=column,
                            suggestion=f"Tier {tier_no} AI term - consider a plainer word.",
                            weight=weight,
                        )
                    )
                continue
            for m in re.finditer(rf"\b{re.escape(word)}\b", lowered):
                line, column = _line_col(text, m.start())
                findings.append(
                    Finding(
                        detector_id=7,
                        detector="AI vocabulary",
                        category="language",
                        match=word,
                        line=line,
                        column=column,
                        suggestion=f"Tier {tier_no} AI term - consider a plainer word.",
                        weight=weight,
                    )
                )

    findings.sort(key=lambda f: (-f.weight, f.detector_id, f.line))
    return findings


# --------------------------------------------------------------------------
# statistical layer
# --------------------------------------------------------------------------


@dataclass
class Stats:
    """Descriptive statistics for a passage."""

    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    mean_sentence_length: float = 0.0
    stdev_sentence_length: float = 0.0
    burstiness: float = 0.0
    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    punctuation_entropy: float = 0.0
    avg_words_per_sentence: float = 0.0
    avg_sentences_per_paragraph: float = 0.0
    comma_rate: float = 0.0
    em_dash_rate: float = 0.0
    passive_marker_rate: float = 0.0
    long_word_rate: float = 0.0
    reading_ease: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[\w'-]+\b")


def _flesch_reading_ease(words: list[str], sentences: int) -> float:
    """Flesch Reading Ease, clamped to a sane range."""
    if not words or not sentences:
        return 0.0
    syllable_count = 0
    for w in words:
        wl = w.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in wl:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if wl.endswith("e") and count > 1:
            count -= 1
        syllable_count += max(count, 1)
    syllables_per_word = syllable_count / len(words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * syllables_per_word
    return round(max(0.0, min(100.0, score)), 2)


def compute_stats(text: str) -> Stats:
    """Compute the statistical profile of ``text``."""
    if not text or not text.strip():
        return Stats()

    words = _WORD.findall(text)
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    sentences = raw_sentences or [text.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    sent_lengths = [len(_WORD.findall(s)) for s in sentences]
    sent_lengths = [n for n in sent_lengths if n > 0] or [len(words)]

    mean_len = statistics.fmean(sent_lengths)
    stdev_len = statistics.pstdev(sent_lengths) if len(sent_lengths) > 1 else 0.0
    # Burstiness: coefficient of variation of sentence length. Human prose
    # sits around 0.35-0.55; heavily regular LLM prose sits below 0.25.
    burstiness = (stdev_len / mean_len) if mean_len else 0.0

    lowered = [w.lower() for w in words]
    unique = len(set(lowered))
    ttr = unique / len(lowered) if lowered else 0.0
    counts: dict[str, int] = {}
    for w in lowered:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = hapax / unique if unique else 0.0

    # Punctuation entropy over , . ; : ! ? - ( )
    punct_counts: dict[str, int] = {}
    for ch in text:
        if ch in ",.;:!?-()":
            punct_counts[ch] = punct_counts.get(ch, 0) + 1
    total_punct = sum(punct_counts.values())
    if total_punct:
        entropy = -sum(
            (c / total_punct) * math.log2(c / total_punct) for c in punct_counts.values()
        )
    else:
        entropy = 0.0

    comma_rate = text.count(",") / len(words) if words else 0.0
    em_dash_rate = text.count("—") / len(words) if words else 0.0

    passive_markers = len(
        re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", text, re.I)
    )
    passive_marker_rate = passive_markers / len(words) if words else 0.0

    long_words = sum(1 for w in words if len(w) >= 8)
    long_word_rate = long_words / len(words) if words else 0.0

    return Stats(
        word_count=len(words),
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs),
        mean_sentence_length=round(mean_len, 2),
        stdev_sentence_length=round(stdev_len, 2),
        burstiness=round(burstiness, 4),
        type_token_ratio=round(ttr, 4),
        hapax_ratio=round(hapax_ratio, 4),
        punctuation_entropy=round(entropy, 4),
        avg_words_per_sentence=round(mean_len, 2),
        avg_sentences_per_paragraph=(
            round(len(sentences) / len(paragraphs), 2) if paragraphs else 0.0
        ),
        comma_rate=round(comma_rate, 4),
        em_dash_rate=round(em_dash_rate, 4),
        passive_marker_rate=round(passive_marker_rate, 4),
        long_word_rate=round(long_word_rate, 4),
        reading_ease=_flesch_reading_ease(words, len(sentences)),
    )


# --------------------------------------------------------------------------
# combined report
# --------------------------------------------------------------------------


@dataclass
class Report:
    """The full analysis of a passage."""

    text: str = ""
    findings: list[Finding] = field(default_factory=list)
    #: Patterns that only appear at the scale of a paragraph or document.
    document_findings: list[DocumentFinding] = field(default_factory=list)
    #: Findings from the threshold-based detectors.
    counted_findings: list[CountFinding] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    pattern_score: float = 0.0
    style_score: float = 0.0
    overall_score: float = 0.0
    band: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_score": round(self.pattern_score, 4),
            "style_score": round(self.style_score, 4),
            "overall_score": round(self.overall_score, 4),
            "band": self.band,
            "stats": self.stats.as_dict(),
            "findings": [f.as_dict() for f in self.findings],
            "finding_count": len(self.findings),
            "counted_findings": [
                {
                    "detector": f.detector,
                    "match": f.match,
                    "line": f.line,
                    "suggestion": f.suggestion,
                    "weight": f.weight,
                }
                for f in self.counted_findings
            ],
            "document_findings": [
                {
                    "detector": f.detector,
                    "match": f.match,
                    "line": f.line,
                    "suggestion": f.suggestion,
                    "weight": f.weight,
                }
                for f in self.document_findings
            ],
            "detector_counts": self.detector_counts(),
        }

    def detector_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.detector] = counts.get(f.detector, 0) + 1
        for f in self.document_findings:
            counts[f.detector] = counts.get(f.detector, 0) + 1
        for f in self.counted_findings:
            counts[f.detector] = counts.get(f.detector, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def top_findings(self, limit: int = 10) -> list[Finding]:
        return self.findings[:limit]


def _pattern_score(
    findings: list[Finding],
    word_count: int,
    document_findings: list[DocumentFinding] | None = None,
    counted_findings: list[CountFinding] | None = None,
) -> float:
    """Density of weighted pattern hits, normalised per 100 words.

    A short text with one hit should not look as bad as a long text with
    one hit, so the raw weight is scaled by length before it saturates.

    Document-scale findings count at half weight: they are weaker signals
    on their own, but several together are meaningful.
    """
    if not word_count:
        return 0.0
    total_weight = sum(f.weight for f in findings)
    if document_findings:
        total_weight += sum(f.weight for f in document_findings) * 0.5
    if counted_findings:
        total_weight += sum(f.weight for f in counted_findings)
    if not total_weight:
        return 0.0
    per_100 = (total_weight / word_count) * 100
    # Saturating curve: 40 weighted points per 100 words is very bad.
    return round(min(1.0, per_100 / 40.0), 4)


def _style_score(stats: Stats) -> float:
    """How uniform is the prose?

    Each component scores 0 (human-like) to 1 (uniform, machine-like).
    """
    if not stats.word_count:
        return 0.0

    # Burstiness: humans 0.35-0.55, LLM often below 0.30.
    if stats.burstiness < 0.25:
        burst = min(1.0, (0.35 - stats.burstiness) / 0.25)
    elif stats.burstiness > 0.85:
        burst = 0.0
    else:
        burst = 0.0

    # Type-token ratio: 0.38-0.55 is normal for prose; lower is repetitive.
    ttr = (
        min(1.0, (0.45 - stats.type_token_ratio) / 0.20)
        if stats.type_token_ratio < 0.40
        else 0.0
    )

    # Hapax ratio: a healthy share of words appear once.
    hapax = (
        min(1.0, (0.45 - stats.hapax_ratio) / 0.30) if stats.hapax_ratio < 0.45 else 0.0
    )

    # Punctuation entropy: varied punctuation gives a higher value.
    if stats.punctuation_entropy < 1.6:
        punct = min(1.0, (1.8 - stats.punctuation_entropy) / 1.2)
    else:
        punct = 0.0

    # Em dash density: one em dash per 100 words is already a lot.
    em = (
        min(1.0, (stats.em_dash_rate - 0.01) / 0.03) if stats.em_dash_rate > 0.01 else 0.0
    )

    # Paragraph uniformity.
    if stats.avg_sentences_per_paragraph:
        if stats.avg_sentences_per_paragraph > 6:
            struct = min(1.0, (stats.avg_sentences_per_paragraph - 6) / 6)
        elif stats.avg_sentences_per_paragraph < 1.6:
            struct = 0.0
        else:
            struct = 0.0
    else:
        struct = 0.0

    components = [burst, ttr, hapax, punct, em, struct]
    weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    return round(sum(c * w for c, w in zip(components, weights, strict=True)), 4)


def _band(score: float) -> str:
    if score < 0.15:
        return "low - reads naturally"
    if score < 0.35:
        return "moderate - some AI tells"
    if score < 0.60:
        return "high - strong AI signature"
    return "very high - pervasive AI signature"


def analyze(text: str) -> Report:
    """Analyse ``text`` and return the full report.

    The score is a style estimate. It is useful for comparing a draft
    against its own rewrite; it is not evidence about authorship.
    """
    # A watched phrase inside a quotation, a title, a code span or a
    # citation is not the writer using it - it is the writer discussing
    # it. Those matches are suppressed before anything is scored.
    findings = filter_findings(detect_patterns(text), text)
    doc_findings = detect_document_patterns(text)
    counted = detect_counted_patterns(text, data.TIER_2, data.TIER_3)
    stats = compute_stats(text)
    pattern_score = _pattern_score(
        findings, stats.word_count, doc_findings, counted
    )
    style_score = _style_score(stats)
    # The lexical layer dominates: a text that says "delve into the
    # intricate tapestry" reads as AI regardless of its sentence rhythm.
    overall = round(0.65 * pattern_score + 0.35 * style_score, 4)
    return Report(
        text=text,
        findings=findings,
        document_findings=doc_findings,
        counted_findings=counted,
        stats=stats,
        pattern_score=pattern_score,
        style_score=style_score,
        overall_score=overall,
        band=_band(overall),
    )
