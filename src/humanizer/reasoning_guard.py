"""Detect a model's internal reasoning masquerading as an answer.

MiniMax-M3 is a reasoning model. With ``reasoning_split`` enabled the
reasoning normally arrives in a separate field, and the client refuses
to use it. But the failure mode is real and was observed in practice:
when a response's content field carries the model's scratchpad - for
instance when the request hit its token limit while the model was still
thinking - the client's extraction returns reasoning and the pipeline
writes the model's deliberation into the user's document.

The result is not cosmetic. The scratchpad contains the prompt, the
placeholder tokens, and the model's commentary on the task. It is a
disclosure of the instruction set, and it is 7x longer than the text it
was supposed to replace.

So the pipeline checks the answer before accepting it.

Detection is tiered rather than density-based. A phrase like "we need
to" occurs in perfectly good prose, so matching on phrases gives false
positives that would silently stop the tool from rewriting legitimate
text. But some markers essentially never appear in a rewritten passage:

* placeholder tokens (``P16``, ``P12``) - these are this pipeline's
  internal placeholders and appear nowhere in user text
* "citation integrity", "forced triad" - the model's commentary on its
  own instructions
* "per the prompt", "the evaluator will" - the model describing the
  instructions it was given

One of those is enough on its own. Weaker markers need to agree with
another signal before they fire.

The cost is asymmetric, so the thresholds are conservative:

* a false positive costs one chunk (the original is kept) - annoying
  but safe
* a false negative costs the user's document

So a short input is never flagged, and weak signals need two to agree.
"""

from __future__ import annotations

import re

#: Markers that essentially never appear in a rewritten passage. One
#: hit is sufficient to reject the response.
_STRONG_META = re.compile(
    r"(?:\bP\d{1,2}\b|"
    r"forced triad|"
    r"citation integrity|"
    r"per the (?:prompt|instruction)\b|"
    r"the (?:evaluator|developer) (?:will|likely|may)\b|"
    r"(?:the|this) (?:prompt|instruction)s? (?:says|asks|instructs|requires)\b|"
    r"the system prompt)",
    re.IGNORECASE,
)

#: Markers that appear in the scratchpad but could plausibly appear in
#: an essay about language models. These need a second signal.
#: "placeholder" is a real English word and appears in legitimate prose
#: ("a placeholder value stands in for..."), so it only counts when it
#: sits alongside another marker - typically with the pipeline's own
#: ``P16``-style placeholder tokens.
_PLACEHOLDER = re.compile(r"placeholder(?:s)?\b", re.IGNORECASE)

_WEAK_META = re.compile(
    r"(?:the (?:user|model) (?:asks|wants|expects)\b|"
    r"check all placeholders\b|"
    r"in same order\b|"
    r"no missing\b|"
    r"refs?\s+P\d|"
    r"\bexact\.\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

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

    lines = [ln for ln in text.split("\n") if ln.strip()]
    # Too short to classify reliably. A one-line answer is not worth
    # discarding on the strength of a single marker.
    if len(lines) < 3:
        return False

    if _STRONG_META.search(text):
        return True

    if _PLACEHOLDER.search(text) and re.search(r"\bP\d{1,2}\b", text):
        return True

    planning_lines = [ln for ln in lines if _PLANNING.match(ln)]
    planning_density = len(planning_lines) / len(lines)

    weak_hits = len(_WEAK_META.findall(text))
    diff_lines = sum(1 for ln in lines if _DIFF.match(ln))
    diff_density = diff_lines / len(lines)

    # Prose has sentence endings. The scratchpad mostly does not.
    sentence_ends = len(_SENTENCE_END.findall(text))
    prose_ratio = sentence_ends / max(1, len(lines))

    signals = 0

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
