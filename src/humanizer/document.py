"""Document-scale patterns from the blader/humanizer catalogue.

The engine in :mod:`humanizer.analyze` catches phrase-level and formatting
tells. This module covers the patterns that only appear at the scale of a
whole paragraph or document — the ones a regex over a single sentence
cannot see:

* a one-line paragraph that restates the paragraph before it
* a contrast split across two sentences instead of one
* several sentences opening the same way
* a heading repeated in the first sentence after it
* a paragraph of parallel three-part constructions
* a staged run-up before the point ("Let's dive in")

Each detector returns the offending text and a short description of what
to do about it. The catalogue's own guardrails are respected: a pattern
that is weak on its own is reported with a lower weight, and patterns that
only make sense in a conversational context are left alone unless the text
clearly is one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Patterns that are only a tell when several of them appear close
#: together. The catalogue is explicit that "several unrelated rejections
#: in a row are a stronger sign than one".
_REPETITION_THRESHOLD = 2


@dataclass(frozen=True)
class DocumentFinding:
    """One document-scale pattern occurrence."""

    detector: str
    match: str
    line: int
    suggestion: str
    weight: int


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Return (offset, paragraph) pairs for the non-empty paragraphs."""
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"(?:^|\n\s*\n)([^\n].*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
        body = m.group(1).strip()
        if body:
            out.append((m.start(1), body))
    return out


def _sentences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        body = m.group(0).strip()
        if body:
            out.append((m.start(), body))
    return out


# --------------------------------------------------------------------------
# individual detectors
# --------------------------------------------------------------------------


def _one_line_closers(text: str) -> list[DocumentFinding]:
    """A one-sentence paragraph restating the paragraph before it."""
    findings: list[DocumentFinding] = []
    paras = _paragraphs(text)
    for i in range(1, len(paras)):
        offset, body = paras[i]
        sentences = _sentences(body)
        if len(sentences) != 1:
            continue
        prev = paras[i - 1][1]
        prev_words = set(re.findall(r"[a-z]+", prev.lower()))
        this_words = set(re.findall(r"[a-z]+", body.lower()))
        if not this_words:
            continue
        # A closer restates the paragraph before it: most of its content
        # words already appeared.
        overlap = len(this_words & prev_words) / len(this_words)
        if overlap >= 0.6 and len(body) < 220:
            findings.append(
                DocumentFinding(
                    detector="One-line closer",
                    match=body[:110],
                    line=_line_of(text, offset),
                    suggestion="Delete it, or fold its point into the paragraph above.",
                    weight=3,
                )
            )
    return findings


_CONTRAST_ACROSS_SENTENCES = re.compile(
    # "This does not mean X. It means Y." - the two halves may be separated
    # by a line break, and the second half's subject may vary.
    r"([^.!?]{10,160})[.!?]\s*(This|That|It)\s+(does not|doesn't|is not|isn't)\s+"
    r"(mean|make|imply|suggest|represent)\b([^.!?]{0,120})[.!?]\s*"
    r"(It|This|That)\s+(means|means that|is about|is really|points to|shows)\b",
    re.IGNORECASE | re.DOTALL,
)


def _contrast_split(text: str) -> list[DocumentFinding]:
    """"This does not mean X. It means Y." across two sentences."""
    findings = []
    for m in _CONTRAST_ACROSS_SENTENCES.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Contrast split across sentences",
                match=" ".join(m.group(0).split())[:110],
                line=_line_of(text, m.start()),
                suggestion="Merge into one sentence: 'X does not mean A; it means B.'",
                weight=3,
            )
        )
    return findings


_STAGED_OPENER = re.compile(
    r"(?:^|\n)\s*(?:Let's|Let us)\s+(?:dive in|explore|break this down|dive into)|"
    r"(?:^|\n)\s*Here's what you need to know|(?:^|\n)\s*Without further ado|"
    r"(?:^|\n)\s*Now let's look at|(?:^|\n)\s*Let's be honest|(?:^|\n)\s*Real talk",
    re.IGNORECASE,
)


def _staged_runup(text: str) -> list[DocumentFinding]:
    """A staged run-up before the point."""
    findings = []
    for m in _STAGED_OPENER.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Staged run-up",
                match=" ".join(m.group(0).split())[:80],
                line=_line_of(text, m.start()),
                suggestion="Start with the point itself.",
                weight=4,
            )
        )
    return findings


#: The boundary may be a sentence end, a line end, the start of the text,
#: or a comma - "To be clear, I am not saying ..." puts two of these
#: phrases in one sentence.
_ARGUING_WITH_NO_ONE = re.compile(
    r"(?:^|[.!?]\s+|\n\s*|,\s+)(?:"
    r"This isn't (?:mainly )?about|"
    r"(?:I(?:'m| am) not saying)|"
    r"To be clear|"
    r"(?:Don't|Do not) get me wrong|"
    r"This is not to say|"
    r"Some might say|"
    r"A tempting approach would be|"
    r"One might be tempted to|"
    r"An obvious approach would be|"
    r"You might think|"
    r"It would be easy to just)",
    re.IGNORECASE,
)


def _arguing_with_no_one(text: str) -> list[DocumentFinding]:
    """Several unrelated rejections in a row - a strong sign."""
    matches = list(_ARGUING_WITH_NO_ONE.finditer(text))
    if len(matches) < _REPETITION_THRESHOLD:
        return []
    return [
        DocumentFinding(
            detector="Arguing with no one",
            match=" ".join(m.group(0).split())[:80],
            line=_line_of(text, m.start()),
            suggestion="Cut the preamble and make the claim directly.",
            weight=4,
        )
        for m in matches
    ]


_SENTENCE_OPENING = re.compile(r"^\s*([A-Za-z]+)")


def _repeated_openings(text: str) -> list[DocumentFinding]:
    """Several consecutive sentences starting the same way.

    The catalogue is explicit that deliberate repetition is allowed
    ("She came. She saw. She conquered."), so this is a soft signal: it
    is reported, but at a low weight.
    """
    findings = []
    # The opening word of each sentence, in order.
    openings: list[tuple[int, str]] = []
    for offset, sentence in _sentences(text):
        m = _SENTENCE_OPENING.match(sentence)
        if m:
            openings.append((offset, m.group(1).lower()))

    run: list[tuple[int, str]] = []
    for item in openings:
        if run and run[-1][1] == item[1]:
            run.append(item)
            continue
        if len(run) >= 3:
            findings.append(
                DocumentFinding(
                    detector="Repeated sentence openings",
                    match=f'{len(run)} consecutive sentences start with "{run[0][1]}"',
                    line=_line_of(text, run[0][0]),
                    suggestion="Vary the openings - the catalogue explicitly allows "
                    "deliberate repetition, so this is a soft signal.",
                    weight=2,
                )
            )
        run = [item]
    if len(run) >= 3:
        findings.append(
            DocumentFinding(
                detector="Repeated sentence openings",
                match=f'{len(run)} consecutive sentences start with "{run[0][1]}"',
                line=_line_of(text, run[0][0]),
                suggestion="Vary the openings - the catalogue explicitly allows "
                "deliberate repetition, so this is a soft signal.",
                weight=2,
            )
        )
    return findings


_TRIAD = re.compile(
    r"\b([A-Za-z][\w'-]{2,}),\s+([A-Za-z][\w'-]{2,}),\s+and\s+([A-Za-z][\w'-]{2,})\b"
)


def _forced_triads(text: str) -> list[DocumentFinding]:
    """Three parallel items, or three parallel sentences."""
    findings = []
    for m in _TRIAD.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Forced triad",
                match=m.group(0)[:80],
                line=_line_of(text, m.start()),
                suggestion="Use two, or let the third carry its own sentence.",
                weight=2,
            )
        )
    # Three short parallel sentences in a row.
    paras = _paragraphs(text)
    for _offset, body in paras:
        sentences = _sentences(body)
        if len(sentences) < 3:
            continue
        for i in range(len(sentences) - 2):
            triple = sentences[i:i + 3]
            lengths = [len(s[1].split()) for s in triple]
            if max(lengths) - min(lengths) > 4:
                continue
            if all(s[1].rstrip(".!?").endswith(("." , "!", "?")) for s in triple):
                findings.append(
                    DocumentFinding(
                        detector="Forced triad",
                        match=" | ".join(s[1][:40] for s in triple),
                        line=_line_of(text, triple[0][0]),
                        suggestion="Three parallel sentences of the same shape - "
                        "break the symmetry.",
                        weight=2,
                    )
                )
    return findings


def _heading_repeated(text: str) -> list[DocumentFinding]:
    """A heading followed by a line that just restates it."""
    findings = []
    pattern = re.compile(
        r"(?m)^(#{1,6})\s+(.+?)\n\s*\n\s*([^.!?\n]{5,120})[.!]?\s*$"
    )
    for m in pattern.finditer(text):
        heading = m.group(2).strip()
        line = m.group(3).strip()
        heading_words = set(re.findall(r"[a-z]+", heading.lower()))
        line_words = set(re.findall(r"[a-z]+", line.lower()))
        if not heading_words or not line_words:
            continue
        overlap = len(heading_words & line_words) / len(heading_words)
        if overlap >= 0.6:
            findings.append(
                DocumentFinding(
                    detector="Heading repeated in the first line",
                    match=f"{heading} -> {line}"[:110],
                    line=_line_of(text, m.start()),
                    suggestion="Say something the heading does not already say.",
                    weight=3,
                )
            )
    return findings


def _deep_closers(text: str) -> list[DocumentFinding]:
    """Sayings that sound deep - the one-line closer family."""
    findings = []
    pattern = re.compile(
        r"(?:^|\.\s+)(?:That(?:'s| is) the real|That distinction matters|"
        r"Read that again|Let that sink in|The message was clear|"
        r"It was a lesson in|At its core,|The heart of the matter|"
        r"The real question is|In reality,|What really matters)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Closing line that sounds deep",
                match=" ".join(m.group(0).split())[:80],
                line=_line_of(text, m.start()),
                suggestion="End on the last concrete fact instead.",
                weight=3,
            )
        )
    return findings


def _writing_about_the_document(text: str) -> list[DocumentFinding]:
    """Describing how the text was assembled rather than its subject."""
    findings = []
    pattern = re.compile(
        r"(?:^|\.\s+)(?:The table below compares|This section is organized by|"
        r"This document (?:was|has been) |"
        r"Here is (?:a|the) (?:summary|overview|breakdown)|"
        r"was added to replace|compiled from)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Writing about the document",
                match=" ".join(m.group(0).split())[:80],
                line=_line_of(text, m.start()),
                suggestion="Cut the account of how you worked; keep the content.",
                weight=2,
            )
        )
    return findings


def _hyphenated_pairs(text: str) -> list[DocumentFinding]:
    """Hyphenated pairs used where a plain word would do."""
    findings = []
    pattern = re.compile(
        r"\b(?:high-quality|well-known|well-documented|long-term|real-time|"
        r"client-facing|results-driven|forward-looking|value-driven|"
        r"best-in-class|world-class|next-generation|future-proof)\b",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        findings.append(
            DocumentFinding(
                detector="Hyphenated pair",
                match=m.group(0),
                line=_line_of(text, m.start()),
                suggestion="A soft signal on its own - plain words usually read "
                "better.",
                weight=1,
            )
        )
    return findings


#: The detectors, in the order they are reported.
DETECTORS = (
    _staged_runup,
    _arguing_with_no_one,
    _one_line_closers,
    _contrast_split,
    _deep_closers,
    _heading_repeated,
    _forced_triads,
    _repeated_openings,
    _writing_about_the_document,
    _hyphenated_pairs,
)


def detect_document_patterns(text: str) -> list[DocumentFinding]:
    """Run every document-scale detector over ``text``."""
    if not text or not text.strip():
        return []
    findings: list[DocumentFinding] = []
    for detector in DETECTORS:
        try:
            findings.extend(detector(text))
        except Exception:  # noqa: BLE001 - a broken detector must not break the scan
            continue
    findings.sort(key=lambda f: (-f.weight, f.line))
    return findings
