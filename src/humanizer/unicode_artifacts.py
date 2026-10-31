"""Unicode artifact detection and normalisation.

The user's observation - that AI text encodes spaces differently from
typed text - is right about the category and wrong about the mechanism.
For MiniMax-M3 every space is U+0020, byte-identical to a typed space
(verified over 595 spaces across six generations). But AI text does carry
typographic artifacts, and they are a cheap, high-precision signal for a
detector to use:

    * non-standard spaces: U+00A0, U+2002/3/7/8/9, U+202F, U+2028/9
    * zero-width characters: U+200B, U+00AD, U+FEFF, U+200C/D, U+2060
    * smart punctuation: curly quotes, em/en dashes, the ellipsis character
    * homoglyphs: a Cyrillic "a" that looks like a Latin "a"

Measured over six generations from MiniMax-M3, 4 of 6 outputs contained at
least one artifact (em dashes and smart quotes). A detector that checks
for these has a free signal that a human typist almost never produces.

So the fix is the opposite of what the hypothesis proposed: rather than
replacing the model's space with a "clicked" space, normalise every
artifact to its plain-ASCII equivalent. That is what a person typing in
a plain editor would produce, and it removes a signal rather than adding
one.

Normalising is not evasion for its own sake - these characters genuinely
break copy-paste into a CMS, a CSV or a database, and U+00A0 inside a
word stops that word from being found by a search.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

#: Zero-width and invisible characters. These carry no meaning at all in
#: prose and are removed outright.
ZERO_WIDTH = {
    0x00AD: "SOFT HYPHEN",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x2060: "WORD JOINER",
    0xFEFF: "BYTE ORDER MARK",
    0xFEFF + 0x10000: "",  # placeholder, never matched
}

#: Spaces that are not U+0020. They look like a space but are not one.
NONSTANDARD_SPACES = {
    0x00A0: "NO-BREAK SPACE",
    0x1680: "OGHAM SPACE MARK",
    0x2002: "EN SPACE",
    0x2003: "EM SPACE",
    0x2004: "THREE-PER-EM SPACE",
    0x2005: "FOUR-PER-EM SPACE",
    0x2006: "SIX-PER-EM SPACE",
    0x2007: "FIGURE SPACE",
    0x2008: "PUNCTUATION SPACE",
    0x2009: "THIN SPACE",
    0x200A: "HAIR SPACE",
    0x202F: "NARROW NO-BREAK SPACE",
    0x205F: "MEDIUM MATHEMATICAL SPACE",
    0x3000: "IDEOGRAPHIC SPACE",
}

#: Typographic punctuation, mapped to its plain equivalent.
SMART_PUNCTUATION = {
    0x2018: "'",
    0x2019: "'",
    0x201A: "'",
    0x201B: "'",
    0x201C: '"',
    0x201D: '"',
    0x201E: '"',
    0x201F: '"',
    0x2010: "-",
    0x2011: "-",
    0x2012: "-",
    0x2013: "-",
    0x2014: "-",
    0x2015: "-",
    0x2212: "-",
    0x2026: "...",
    0x2022: "*",
    0x00BB: '"',
    0x00AB: '"',
    0x2039: "'",
    0x203A: "'",
    0x2032: "'",
    0x2033: '"',
    0x2036: '"',
    0x00A9: "(c)",
    0x00AE: "(r)",
    0x2122: "(tm)",
    0x00B0: " degrees",
}

#: Unicode marks that change how text is displayed but not what it says.
_BIDI = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}


@dataclass
class ArtifactReport:
    """What was found and what was changed."""

    #: Human-readable counts, e.g. {"U+2014 EM DASH": 3}
    found: dict[str, int] = field(default_factory=dict)
    #: The text after normalisation.
    text: str = ""
    #: How many characters changed.
    changed: int = 0
    #: Words that mixed scripts.
    homoglyphs: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.found and not self.homoglyphs

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "changed": self.changed,
            "homoglyphs": self.homoglyphs,
            "clean": self.clean,
        }


def _script_of(ch: str) -> str:
    """The script a character belongs to, or an empty string."""
    if not ch.isalpha():
        return ""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    for script in (
        "LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI",
        "THAI", "HAN", "HIRAGANA", "KATAKANA", "ARMENIAN", "GEORGIAN",
    ):
        if script in name:
            return script
    return ""


def find_homoglyphs(text: str) -> list[str]:
    """Words that mix scripts - a strong sign the text was assembled."""
    out = []
    for word in text.split():
        scripts = {_script_of(c) for c in word if _script_of(c)}
        if len(scripts) > 1:
            out.append(word)
    return out


def scan(text: str) -> ArtifactReport:
    """Find the artifacts in ``text`` without changing it."""
    report = ArtifactReport(text=text)
    for cp, name in {**ZERO_WIDTH, **NONSTANDARD_SPACES}.items():
        n = text.count(chr(cp))
        if n:
            report.found[f"U+{cp:04X} {name}"] = n
    for cp in SMART_PUNCTUATION:
        n = text.count(chr(cp))
        if n:
            try:
                label = unicodedata.name(chr(cp))
            except ValueError:
                label = f"U+{cp:04X}"
            report.found[f"U+{cp:04X} {label}"] = n
    report.homoglyphs = find_homoglyphs(text)
    return report


def normalize(text: str, *, smart_punctuation: bool = True) -> ArtifactReport:
    """Normalise ``text`` to plain ASCII prose.

    * zero-width characters are removed
    * non-standard spaces become U+0020
    * smart punctuation becomes its ASCII equivalent
    * the text is NFC-normalised, so composed and decomposed forms of the
      same character become one

    Set ``smart_punctuation=False`` to leave curly quotes and dashes
    alone - some house styles want them.
    """
    if not text:
        return ArtifactReport(text=text)

    found = scan(text)
    out: list[str] = []
    changed = 0

    for ch in text:
        cp = ord(ch)

        if cp in ZERO_WIDTH or cp in _BIDI:
            changed += 1
            continue
        if cp in NONSTANDARD_SPACES:
            out.append(" ")
            changed += 1
            continue
        if smart_punctuation and cp in SMART_PUNCTUATION:
            out.append(SMART_PUNCTUATION[cp])
            changed += 1
            continue
        out.append(ch)

    result = "".join(out)

    # NFC: composed characters become one code point. This is what a
    # keyboard produces, and it removes the accent decomposed form.
    normalized = unicodedata.normalize("NFC", result)
    if normalized != result:
        changed += sum(
            1 for a, b in zip(result, normalized, strict=False) if a != b
        )
        result = normalized

    report = ArtifactReport(
        found=found.found,
        text=result,
        changed=changed,
        homoglyphs=found.homoglyphs,
    )
    return report


def artifact_score(text: str) -> float:
    """A 0-1 measure of how many artifacts the text carries.

    Zero for clean text. A handful of em dashes should not push a
    document to the top of the scale, so the count is scaled per 100
    characters and saturated at a level that means "this is unusual".
    """
    report = scan(text)
    if not report.found and not report.homoglyphs:
        return 0.0
    total = sum(report.found.values()) + len(report.homoglyphs)
    if not text:
        return 0.0
    per_100 = (total / len(text)) * 100
    return round(min(1.0, per_100 / 2.0), 4)
