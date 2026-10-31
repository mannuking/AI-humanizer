"""Context guardrails: where a detector must NOT fire.

The reference catalogue is explicit about this, and it is the difference
between a detector that reads like a person and one that does not:

    "Leave a watched phrase alone inside a quotation, a title, a proper
    name, or a passage that discusses the phrase rather than uses it."

A human who writes *"delve into the intricate tapestry"* in order to
criticise that phrase has not written AI-sounding prose. Flagging it is
a false positive, and a detector that produces false positives gets
ignored.

This module finds the regions where a match should be suppressed:

* quoted text - straight quotes, curly quotes, and block quotes
* inline code and code fences
* markdown headings - a title is not prose
* link and image targets - the URL is not the caption
* reference-style citation markers

Everything else is fair game.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Guard:
    """A span of text where detector matches should be suppressed."""

    start: int
    end: int
    kind: str


#: Straight double quotes, curly double quotes, and single quotes.
#: The curly forms are what an editor's auto-correct produces, so they
#: are the common case in real text.
_QUOTE_PAIRS = (
    ('"', '"', "quotation"),
    ("\u201c", "\u201d", "quotation"),
    ("\u2018", "\u2019", "quotation"),
    ("'", "'", "quotation"),
)

#: A pair of quoted spans on one line. This deliberately does not span
#: lines: a quote that opens on one line and closes ten lines later is
#: usually a block quote, and the block-quote rule catches that.
_INLINE_QUOTE = re.compile(
    r'"[^"\n]{1,600}"'          # straight double quotes
    r"|\u201c[^\u201d\n]{1,600}\u201d"   # curly double quotes
    r"|\u2018[^\u2019\n]{1,600}\u2019"   # curly single quotes
)

#: A single-quoted span. Apostrophes make this unreliable, so it is
#: conservative: the quote must be preceded by a space or an opening
#: bracket, and must not look like a contraction.
_SINGLE_QUOTE = re.compile(r"(?:(?<=\s)|(?<=\())'([^'\n]{2,300})'(?=\s|[.,;:!?)]|$)")

#: ``` fenced code blocks ``` and ``` inline code ```
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]{1,500}`")

#: Markdown headings, and setext headings (underlined with === or ---).
_ATX_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+\S.*$")
_SETEXT_HEADING = re.compile(r"(?m)^[^\n]{1,200}\n[=-]{3,}\s*$")

#: The target of a markdown link or image: [caption](target)
_LINK_TARGET = re.compile(r"\]\([^)\n]{1,500}\)")

#: Reference-style markers: [1], [12], [Smith 2023]
_REF_MARKER = re.compile(r"\[[^\]\n]{1,40}\]")

#: A block quote: every line starting with '>'
_BLOCK_QUOTE = re.compile(r"(?m)(?:^[ \t]*>.*(?:\n|$))+")

#: Salutations and sign-offs. These predate chatbots, so a "delve" in
#: "Dear Dr. Smith," is not a tell.
_SALUTATION = re.compile(
    r"(?mi)^[ \t]*(?:dear|hi|hello|hey|greetings|good morning|good afternoon|"
    r"good evening|sincerely|regards|best regards|kind regards|best wishes|"
    r"yours truly|yours sincerely|respectfully|thank you|thanks|cheers)[^\n]{0,80}$"
)


def _spans_from_pairs(text: str) -> list[Guard]:
    """Find quoted regions by matching opening and closing delimiters."""
    guards: list[Guard] = []
    for opener, closer, kind in _QUOTE_PAIRS:
        if opener == closer:
            # A symmetric pair: scan for them in order.
            pos = 0
            while True:
                start = text.find(opener, pos)
                if start < 0:
                    break
                end = text.find(closer, start + 1)
                if end < 0:
                    break
                guards.append(Guard(start, end + 1, kind))
                pos = end + 1
            continue
        pos = 0
        while True:
            start = text.find(opener, pos)
            if start < 0:
                break
            end = text.find(closer, start + 1)
            if end < 0:
                break
            guards.append(Guard(start, end + 1, kind))
            pos = end + 1
    return guards


def _spans_from_pattern(text: str, pattern: re.Pattern[str], kind: str) -> list[Guard]:
    return [Guard(m.start(), m.end(), kind) for m in pattern.finditer(text)]


def find_guarded_spans(text: str) -> list[Guard]:
    """Every region of ``text`` where a detector match should be ignored."""
    if not text:
        return []

    guards: list[Guard] = []

    # Code first: a code fence can contain quotes and headings, and those
    # are code, not prose.
    guards += _spans_from_pattern(text, _CODE_FENCE, "code")
    guards += _spans_from_pattern(text, _INLINE_CODE, "code")

    # Block quotes before inline quotes: a block quote may contain the
    # quote characters, and the whole block is guarded either way.
    guards += _spans_from_pattern(text, _BLOCK_QUOTE, "quotation")

    # Quoted text. Symmetric pairs first, then the conservative single
    # quote, then the inline patterns.
    guards += _spans_from_pairs(text)
    guards += _spans_from_pattern(text, _INLINE_QUOTE, "quotation")
    guards += _spans_from_pattern(text, _SINGLE_QUOTE, "quotation")

    # Titles.
    guards += _spans_from_pattern(text, _ATX_HEADING, "title")
    guards += _spans_from_pattern(text, _SETEXT_HEADING, "title")

    # Link and image targets.
    guards += _spans_from_pattern(text, _LINK_TARGET, "link-target")

    # Reference markers.
    guards += _spans_from_pattern(text, _REF_MARKER, "citation")

    # Salutations and sign-offs.
    guards += _spans_from_pattern(text, _SALUTATION, "salutation")

    return _merge(guards, text)


def _merge(guards: list[Guard], text: str) -> list[Guard]:
    """Merge overlapping guards and drop the ones that are not worth having."""
    if not guards:
        return []

    # A zero-length guard protects nothing.
    guards = [g for g in guards if g.end > g.start]

    # Sort and merge overlaps. When two guards overlap, keep the more
    # specific kind if one of them is more specific.
    guards.sort(key=lambda g: (g.start, -g.end))
    merged: list[Guard] = []
    for g in guards:
        if merged and g.start < merged[-1].end:
            last = merged[-1]
            if g.end > last.end:
                merged[-1] = Guard(last.start, g.end, last.kind)
            continue
        merged.append(g)

    # A guard covering most of the document is not a guardrail, it is a
    # bug in the detector's input. Drop it.
    limit = max(200, int(len(text) * 0.6))
    return [g for g in merged if (g.end - g.start) <= limit]


def guarded_regions(text: str) -> list[tuple[int, int]]:
    """The (start, end) offsets of every guarded region."""
    return [(g.start, g.end) for g in find_guarded_spans(text)]


def is_guarded(text: str, start: int, end: int) -> bool:
    """Whether the span ``[start, end)`` falls inside a guarded region."""
    for g_start, g_end in guarded_regions(text):
        if start >= g_start and end <= g_end:
            return True
        # A match that merely overlaps a guard is also suppressed: a
        # detector pattern that starts before a quote and ends inside it
        # is still matching quoted text.
        if start < g_end and end > g_start:
            return True
    return False


#: Detectors that are never suppressed. "Curly quotes" reports on the
#: quote characters themselves, so guarding a quoted region would hide
#: the very thing it is reporting.
_NEVER_GUARDED = frozenset({"Curly quotes"})


def filter_findings(findings, text: str):
    """Drop the findings whose match falls inside a guarded region.

    ``findings`` is any sequence of objects with a ``match`` attribute and
    a ``detector`` attribute. The returned list keeps the original order.
    """
    if not findings:
        return []
    regions = guarded_regions(text)
    if not regions:
        return list(findings)

    kept = []
    for f in findings:
        if getattr(f, "detector", "") in _NEVER_GUARDED:
            kept.append(f)
            continue
        match = getattr(f, "match", "") or ""
        start = getattr(f, "index", None)
        if start is None:
            # No offset on the finding: find every occurrence of the
            # match text and keep the finding only if NONE of them sits
            # in a guarded region. A phrase can appear once inside a
            # quotation and once outside it, and the outside one is the
            # real signal.
            positions = []
            if match:
                # The detectors match case-insensitively, so the match
                # text may be lower-cased while the text has "Delve".
                # Search case-insensitively to find the real offset.
                haystack = text.lower()
                needle = match.lower()
                pos = haystack.find(needle)
                while pos >= 0:
                    positions.append(pos)
                    pos = haystack.find(needle, pos + 1)
            if not positions:
                kept.append(f)
                continue
            end_len = len(match)
            # A finding survives if the phrase occurs at least once
            # outside every guarded region. "delve into the tapestry"
            # appearing once in a quotation and once in the writer's own
            # prose is still a real hit on the prose.
            unguarded = [
                p
                for p in positions
                if not any(
                    p < g_end and p + end_len > g_start
                    for g_start, g_end in regions
                )
            ]
            if unguarded:
                kept.append(f)
            continue
        end = start + len(match)
        suppressed = any(
            start < g_end and end > g_start for g_start, g_end in regions
        )
        if not suppressed:
            kept.append(f)
    return kept
