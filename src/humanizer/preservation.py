"""Protected spans: the things a rewrite must not touch.

Measured across the commercial humanizers, every one of them corrupts or
drops between 6% and 35% of the numeric tokens in the input - Walter
Writes loses about 19%. That failure is invisible in the marketing and it
is the easiest thing to fix: extract the spans that must survive, hide them
behind opaque placeholders while the model works, then put them back and
*verify* they came back.

The span types below are ordered from most to least likely to be damaged
by a naive rewrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: A number with a unit, a percentage, a currency amount, or a measurement.
#: A number ends on a digit, and the thousands separator may only appear
#: BETWEEN digits. Writing ``[\d,]*`` lets the comma of a following clause
#: be swallowed ("22," instead of "22"), and a lookahead of ``(?!\w)``
#: makes the engine backtrack off a decimal ("$12" instead of "$12.40").
_NUMBER = re.compile(
    r"""
    (?<![\w.])
    (?:
        (?:[$€£¥₹]\s?\d(?:[\d,]*\d)?(?:\.\d+)?)              # currency
      | (?:\d(?:[\d,]*\d)?(?:\.\d+)?\s?%)                     # percentage
      | (?:\d(?:[\d,]*\d)?(?:\.\d+)?\s?°[CF]?)                 # degrees
      | (?:\d(?:[\d,]*\d)?(?:\.\d+)?(?:kg|g|mg|km|cm|mm|m|ml|l|kb|mb|gb|tb|hz|khz|mhz|ghz|w|kw|v|mv|ms|s|min|h|hrs|years?|months?|days?|weeks?)\b)
      | (?:[$€£¥₹]\s?\d(?:[\d,]*\d)?(?:\.\d+)?(?:million|billion|trillion|k|bn|m))
      | (?:\d(?:[\d,]*\d)?(?:\.\d+)?)                          # bare number
    )
    (?!\d)(?!\.\d)
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Dates in the common formats.
_DATE = re.compile(
    r"\b(?:"
    r"(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}"       # 2026-10-31
    r"|\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}"       # 31/10/2026
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+(?:19|20)\d{2}"
    r"|(?:19|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)

#: Version strings, identifiers and chemical/maths notation.
_IDENTIFIER = re.compile(
    r"\b(?:v?\d+\.\d+(?:\.\d+)*(?:-[A-Za-z0-9.]+)?|"
    r"[A-Z]{2,}[-_]\d+[A-Za-z0-9_-]*|"
    r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b"               # CamelCase
    r")\b"
)

#: Emails and URLs.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
#: The trailing character class excludes sentence punctuation, so a URL
#: at the end of a sentence does not swallow the full stop.
_URL = re.compile(
    r"\b(?:https?://|www\.)[^\s)\]<>\"'`]*[A-Za-z0-9/#?=_-]",
    re.IGNORECASE,
)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:\w]+", re.IGNORECASE)

#: Bracketed numeric citations and author-year citations.
_CITATION = re.compile(
    r"\[[\d\s,;–-]{1,40}\]|\((?:[A-Z][A-Za-z'’\-]+\s+et al\.?,?\s*)?(?:19|20)\d{2}[a-z]?\)"
)

#: Markdown emphasis markers wrapped around a phrase.
_EMPHASIS = re.compile(r"\*\*[^*\n]{1,200}\*\*|__[^_\n]{1,200}__")

#: Ordered by how badly a rewrite tends to damage them.
_SPAN_PATTERNS = (
    ("URL", _URL),
    ("EMAIL", _EMAIL),
    ("DOI", _DOI),
    ("CODE", re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]+`", re.DOTALL)),
    ("MATH", re.compile(
        r"\$\$.*?\$\$"
        r"|\\\[[^\]]*\\\]"
        r"|\\\([^)]*\\\)"
        # Inline maths: a single $ pair, but not when a "$" is followed by a
        # digit - that is a price ("$12.40 ... $50.00"), not a formula.
        r"|(?<!\$)\$(?!\$|\s|\d)[^$\n]{1,200}?(?<!\d)\$(?!\$)",
        re.DOTALL,
    )),
    ("TABLE", re.compile(r"(?:^\|.*\|[ \t]*$\n?)+", re.MULTILINE)),
    ("CITATION", _CITATION),
    ("DATE", _DATE),
    ("EMPHASIS", _EMPHASIS),
    # NUMBER before IDENTIFIER: the identifier pattern matches a bare
    # "12.40" and would leave the "$" outside the protected span, where
    # the rewrite could separate it from the amount.
    ("NUMBER", _NUMBER),
    ("IDENTIFIER", _IDENTIFIER),
)


@dataclass
class ProtectedSpans:
    """The protected spans of a text and their placeholders."""

    text: str
    spans: dict[str, str] = field(default_factory=dict)
    #: The original text of each span, for verification.
    originals: list[str] = field(default_factory=list)

    def restore(self, text: str) -> str:
        """Put the spans back, longest placeholder first."""
        out = text
        for token in sorted(self.spans, key=len, reverse=True):
            out = out.replace(token, self.spans[token])
        return out

    def missing(self, text: str) -> list[str]:
        """Which protected values are absent from ``text``?

        Used as a post-condition: a rewrite that dropped a number is a
        failed rewrite, and this says which one.
        """
        absent = []
        for original in self.originals:
            if original not in text:
                absent.append(original)
        return absent


def extract(text: str) -> ProtectedSpans:
    """Replace every protected span in ``text`` with a placeholder.

    Placeholders are deliberately opaque (``⟦P7⟧``) so the model does not
    try to interpret, translate or "improve" them.
    """
    spans: dict[str, str] = {}
    originals: list[str] = []
    working = text
    counter = 0

    for label, pattern in _SPAN_PATTERNS:
        def _swap(match: re.Match, label: str = label) -> str:
            nonlocal counter
            token = f"⟦P{counter}⟧"
            spans[token] = match.group(0)
            originals.append(match.group(0))
            counter += 1
            return token

        working = pattern.sub(_swap, working)

    return ProtectedSpans(text=working, spans=spans, originals=originals)


def strip_placeholders(text: str) -> str:
    """Remove any placeholder the model echoed but failed to keep intact."""
    return re.sub(r"⟦P\d+⟧", "", text)


def verify(input_text: str, output_text: str) -> list[str]:
    """Return the protected values from the input that the output lost."""
    protected = extract(input_text)
    return protected.missing(output_text)
