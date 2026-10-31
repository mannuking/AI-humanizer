"""Detect a model's internal reasoning masquerading as an answer.

MiniMax-M3 is a reasoning model. With ``reasoning_split`` enabled the
reasoning normally arrives in a separate field and the client refuses to
use it. But the failure mode is real and was observed in practice: when
a response's content field carries the model's scratchpad - for
instance when the request hit its token limit while the model was still
thinking - the client's extraction returns reasoning, and the pipeline
writes the model's deliberation into the user's document.

The result is not cosmetic. The scratchpad contains the prompt, the
placeholder tokens, and the model's commentary on the task. It is a
disclosure of the instruction set, and it was 7x longer than the text it
was supposed to replace.

So the pipeline checks the answer before accepting it.

**The placeholder trap.** The pipeline protects numbers, dates, URLs
and citations behind opaque placeholders (``\u27e6P16\u27e7``) before
the model sees the text. The model echoes those placeholders back, so
*every* legitimate rewrite contains tokens like ``P16``. A detector
that keys on placeholder tokens therefore fires on correct output. The
markers below avoid the bare token and look for the commentary around
it instead - "Check all placeholders", "P22 then P14", "in same order" -
which is what scratchpad looks like and rewritten prose does not.

Detection is tiered. A marker that essentially never appears in a
rewritten passage rejects the response on its own; weaker markers need
a second signal to agree.

The cost is asymmetric, so the thresholds are conservative:

* a false positive costs one chunk (the original is kept) - annoying
  but safe
* a false negative costs the user's document

So a short input is never classified.
"""

from __future__ import annotations

import re

#: Markers that essentially never appear in a rewritten passage. One
#: hit is sufficient to reject the response.
_STRONG_META = re.compile(
    r"(?:"
    r"forced triad|"
    r"citation integrity|"
    r"per the (?:prompt|instruction)\b|"
    r"the (?:evaluator|developer) (?:will|likely|may)\b|"
    r"(?:the|this) (?:prompt|instruction)s? (?:says|asks|instructs|requires)\b|"
    r"the system prompt|"
    r"check all placeholders|"
    r"in same order|"
    r"no missing\b|"
    r"refs?\s+P\d)"
    ,
    re.IGNORECASE,
)

#: Markers that appear in the scratchpad but could plausibly appear in
#: an essay about language models. These need a second signal.
_WEAK_META = re.compile(
    r"(?:the (?:user|model) (?:asks|wants|expects)\b|"
    r"maybe they expect|"
    r"need perhaps|"
    r"let's use that|"
    r"we should not alter|"
    r"potential issue\b)",
    re.IGNORECASE,
)

#: A line holding three or more placeholder tokens. Rewritten prose
#: spreads protected values through sentences; the scratchpad lists
#: them.
_PLACEHOLDER_LINE = re.compile(r"(?:\bP\d{1,2}\b.*){3,}", re.IGNORECASE)

#: First-person planning statements. A rewritten passage rarely contains
#: these; the model's scratchpad is full of them.
_PLANNING = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?"
    r"(?:ok(?:ay)?|so|and|but|also|now|then|well|alright|hmm|wait|"
    r"need|maybe|perhaps|check|note|consider|use|keep|try|"
    r"we|i|the model|the user|it)\s+"
    r"(?:need|needs|should|must|can|could|will|would|may|might|have|has|"
    r"want|ask|expect|think|know|see|note|check|use|keep|try|let)\b",
    re.IGNORECASE,
)

#: A line that is a diff or an edit list, not prose.
_DIFF = re.compile(r"^\s*(?:[-+*]\s+[A-Za-z]|\+|-|=>|->)")

#: Prose signals: a rewritten passage has sentences.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(\s|$)")


def looks_like_reasoning(text: str) -> bool:
    """True when ``text`` is the model's scratchpad, not an answer.

    Returns False for anything too short to classify.
    """
    if not text or not text.strip():
        return False

    # Too short to classify reliably. A one-line answer is not worth
    # discarding on the strength of a single marker.
    if len(text.strip()) < 80:
        return False

    # A strong marker rejects on its own, and it is checked before the
    # line-count gate: the model can deliver its scratchpad as a single
    # long line, and a leak that arrives as one paragraph must still be
    # caught.
    if _STRONG_META.search(text):
        return True

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False

    planning_lines = [ln for ln in lines if _PLANNING.match(ln)]
    planning_density = len(planning_lines) / len(lines)

    weak_hits = len(_WEAK_META.findall(text))
    diff_lines = sum(1 for ln in lines if _DIFF.match(ln))
    diff_density = diff_lines / len(lines)

    # Prose has sentence endings. The scratchpad mostly does not.
    sentence_ends = len(_SENTENCE_END.findall(text))
    prose_ratio = sentence_ends / max(1, len(lines))

    signals = 0

    # A line listing three or more placeholder tokens is a token
    # inventory, not a sentence.
    if any(_PLACEHOLDER_LINE.search(ln) for ln in lines):
        signals += 1

    if planning_density >= 0.35:
        signals += 1
    if weak_hits >= 2:
        signals += 1
    if diff_density >= 0.4:
        signals += 1
    if prose_ratio < 0.3:
        signals += 1

    # Two independent signals must agree. One alone is not enough - a
    # list-heavy passage can legitimately have a low prose ratio.
    return signals >= 2
