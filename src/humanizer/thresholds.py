"""Count-based and conditional detectors.

Several of the reference detectors are not "does this regex appear" but
"does this regex appear often enough, or under this condition". Their
regexes are extracted into the catalogue, but the decision logic lives
here, in Python, where the thresholds are readable and testable.

The reference implementation applies these to line-anchored text. The
line semantics are preserved with :data:`re.MULTILINE`.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

#: Words that stay capitalised in a heading and must not count towards
#: the title-case ratio.
SKIP_HEADING_WORDS = re.compile(
    r"^(I|AI|API|CLI|URL|HTML|CSS|JS|TS|NPM|NYC|USA|UK|EU|LLM|GPT|SaaS|IoT|"
    r"CEO|CTO|VP|PR|HR|IT|UI|UX)\b"
)

#: Synonym sets the reference uses to spot a thing being renamed from one
#: sentence to the next.
SYNONYM_SETS: tuple[tuple[str, ...], ...] = (
    ("protagonist", "main character", "central figure", "hero", "lead character", "lead"),
    ("company", "firm", "organization", "enterprise", "corporation", "establishment", "entity"),
    ("city", "metropolis", "urban center", "municipality", "locale", "township"),
    ("building", "structure", "edifice", "facility", "complex", "establishment"),
    ("tool", "instrument", "mechanism", "apparatus", "device", "utility"),
    ("country", "nation", "state", "republic", "sovereign state"),
    ("problem", "challenge", "issue", "obstacle", "hurdle", "difficulty"),
    ("solution", "approach", "methodology", "framework", "strategy", "paradigm"),
)


@dataclass(frozen=True)
class CountFinding:
    """A finding from a threshold-based detector."""

    detector: str
    match: str
    line: int
    suggestion: str
    weight: int
    category: str = "style"


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.MULTILINE))


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------


def title_case_headings(text: str) -> list[CountFinding]:
    """Headings that capitalise most of their words.

    The threshold is the reference one: a heading of three or more words
    where more than 70% of the words start with a capital, ignoring
    acronyms.
    """
    findings = []
    for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE):
        heading = m.group(1).strip()
        words = heading.split()
        if len(words) < 3:
            continue
        capitalised = sum(
            1 for w in words if w[:1].isupper() and not SKIP_HEADING_WORDS.match(w)
        )
        if capitalised / len(words) > 0.7:
            findings.append(
                CountFinding(
                    detector="Title Case headings",
                    match=m.group(0).strip()[:80],
                    line=_line_of(text, m.start()),
                    suggestion="Use sentence case for headings - only the first word "
                    "and proper nouns need a capital.",
                    weight=1,
                )
            )
    return findings


def inline_header_lists(text: str) -> list[CountFinding]:
    """Lists where every item opens with a bold label and a colon."""
    pattern = r"^[*-]\s+\*\*[^*]+:\*\*\s"
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) < 2:
        return []
    return [
        CountFinding(
            detector="Inline-header lists",
            match=f"{len(matches)} list items open with a bold label and a colon",
            line=_line_of(text, matches[0].start()),
            suggestion="Turn the labelled list into prose.",
            weight=3,
        )
    ]


def excessive_structure(text: str) -> list[CountFinding]:
    """More headings or list items than the content warrants."""
    findings = []
    words = len(re.findall(r"\b[\w'-]+\b", text))
    headers = _count(r"^#{1,6}\s+.+$", text)
    bullets = _count(r"^[\s]*[-*+]\s+", text)
    numbered = _count(r"^[\s]*\d+\.\s+", text)
    items = bullets + numbered

    if words < 300 and headers >= 3:
        findings.append(
            CountFinding(
                detector="Excessive structure",
                match=f"{headers} headers in {words} words",
                line=1,
                suggestion="Too many headers for short content - use prose.",
                weight=3,
            )
        )
    if words < 200 and items >= 8:
        findings.append(
            CountFinding(
                detector="Excessive structure",
                match=f"{items} list items in {words} words",
                line=1,
                suggestion="Too many list items for short content - use prose.",
                weight=3,
            )
        )
    return findings


def em_dash_overuse(text: str) -> list[CountFinding]:
    """More than one em dash per 100 words, with at least two of them."""
    em_dashes = text.count("—")
    words = len(re.findall(r"\b[\w'-]+\b", text))
    if not words or em_dashes < 2:
        return []
    ratio = em_dashes / (words / 100)
    if ratio <= 1.0:
        return []
    index = text.find("—")
    return [
        CountFinding(
            detector="Em dash overuse",
            match=f"{em_dashes} em dashes in {words} words",
            line=_line_of(text, max(index, 0)),
            suggestion="Replace most em dashes with commas, periods or parentheses.",
            weight=2,
        )
    ]


def boldface_overuse(text: str) -> list[CountFinding]:
    """Three or more bold spans - bold as a highlighting crutch."""
    matches = list(re.finditer(r"\*\*[^*]+\*\*", text))
    if len(matches) < 3:
        return []
    return [
        CountFinding(
            detector="Boldface overuse",
            match=f"{len(matches)} bold spans",
            line=_line_of(text, matches[0].start()),
            suggestion="Remove the mechanical bold emphasis and let the writing carry it.",
            weight=2,
        )
    ]


def emoji_overuse(text: str) -> list[CountFinding]:
    """Three or more emoji."""
    matches = list(
        re.finditer(
            "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U00002300-\U000023FF\U00002B50]",
            text,
        )
    )
    if len(matches) < 3:
        return []
    return [
        CountFinding(
            detector="Emoji overuse",
            match=f"{len(matches)} emoji",
            line=_line_of(text, matches[0].start()),
            suggestion="Remove emoji decoration from professional text.",
            weight=2,
        )
    ]


def synonym_cycling(text: str) -> list[CountFinding]:
    """The same thing renamed in consecutive sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    findings = []
    for synonyms in SYNONYM_SETS:
        for i in range(len(sentences) - 1):
            first = {w.lower() for w in re.findall(r"[A-Za-z]+", sentences[i])}
            second = {w.lower() for w in re.findall(r"[A-Za-z]+", sentences[i + 1])}
            hits = [
                (a, b)
                for a in synonyms
                for b in synonyms
                if a != b and a in first and b in second
            ]
            if hits:
                a, b = hits[0]
                findings.append(
                    CountFinding(
                        detector="Synonym cycling",
                        match=f'"{a}" then "{b}"',
                        line=1,
                        suggestion="Keep one word for the thing instead of renaming it "
                        "each sentence.",
                        weight=2,
                        category="language",
                    )
                )
                break
    return findings


def tier_vocabulary_overuse(
    text: str,
    tier2: tuple[str, ...],
    tier3: tuple[str, ...],
    total_words: int,
) -> list[CountFinding]:
    """Tier-2 vocabulary appearing twice or more, tier-3 above 3% density.

    These are the reference's thresholds for the two lower tiers. Tier 1
    words are a tell wherever they appear and are handled by the
    lexical detector.
    """
    findings = []
    if not total_words:
        return findings
    lowered = text.lower()

    tier2_hits = []
    for word in tier2:
        count = lowered.count(word.lower())
        if count >= 2:
            tier2_hits.append((word, count))
    if tier2_hits:
        total = sum(c for _w, c in tier2_hits)
        findings.append(
            CountFinding(
                detector="AI vocabulary",
                match=f"{len(tier2_hits)} tier-2 AI terms used {total} times "
                f"(e.g. {', '.join(w for w, _ in tier2_hits[:4])})",
                line=1,
                suggestion="Replace these with plainer words.",
                weight=3,
                category="language",
            )
        )

    tier3_hits = []
    for word in tier3:
        count = lowered.count(word.lower())
        if count:
            tier3_hits.append((word, count))
    tier3_total = sum(c for _w, c in tier3_hits)
    if total_words and tier3_total / total_words > 0.03:
        findings.append(
            CountFinding(
                detector="AI vocabulary",
                match=f"tier-3 AI terms at {tier3_total / total_words:.1%} density",
                line=1,
                suggestion="Vary the wording - these generic words are over-used.",
                weight=2,
                category="language",
            )
        )
    return findings


#: The threshold-based detectors that do not need extra arguments.
DETECTORS = (
    title_case_headings,
    inline_header_lists,
    excessive_structure,
    em_dash_overuse,
    boldface_overuse,
    emoji_overuse,
    synonym_cycling,
)


def detect_counted_patterns(
    text: str,
    tier2: tuple[str, ...] = (),
    tier3: tuple[str, ...] = (),
) -> list[CountFinding]:
    """Run the threshold-based detectors over ``text``."""
    if not text or not text.strip():
        return []
    findings: list[CountFinding] = []
    for detector in DETECTORS:
        try:
            findings.extend(detector(text))
        except Exception:  # noqa: BLE001 - a broken detector must not break the scan
            continue
    if tier2 or tier3:
        total_words = len(re.findall(r"\b[\w'-]+\b", text))
        with contextlib.suppress(Exception):
            findings.extend(tier_vocabulary_overuse(text, tier2, tier3, total_words))
    findings.sort(key=lambda f: (-f.weight, f.line))
    return findings
