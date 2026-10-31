#!/usr/bin/env python
"""Convert the reference JS pattern catalogue into the Python data module.

Reads the JS sources pulled from the reference implementations and emits
``src/humanizer/data.py`` with:

  * TIER_1 / TIER_2 / TIER_3  - the vocabulary blacklists
  * AI_PHRASES                 - regex-based phrase patterns with a fix hint
  * PATTERNS                   - the 30 named detectors with regexes

This runs once; the generated module is committed so the package has no
build-time dependency on the JS sources.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\jk422\humanizer_research\brandonwise__humanizer")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           r"E:\Projects\AI-humanizer\src\humanizer\data.py")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def strip_js_comments(s: str) -> str:
    """Remove JS comments without damaging regex literals.

    A naive block-comment strip is wrong here: the source contains regex
    literals like ``/\*\*[^*]+\*\*/g`` whose text looks like a block
    comment opener, and stripping those silently deletes source code.

    Only block comments that begin a line are removed - that is the
    convention the reference sources follow for their JSDoc blocks.
    """
    s = re.sub(r"^\s*//[^\n]*$", "", s, flags=re.M)
    s = re.sub(r"^\s*/\*.*?\*/", "", s, flags=re.M | re.S)
    return s


def const_block(src: str, name: str) -> str:
    i = src.find(f"const {name}")
    if i < 0:
        return ""
    j = src.find("\n];", i)
    return src[i:j + 3] if j > 0 else ""


def parse_string_array(src: str, name: str) -> list[str]:
    blk = strip_js_comments(const_block(src, name))
    return [x for x in re.findall(r"'([^']*)'", blk) if x.strip()]


# A JS regex literal: /body/flags. The body may contain escaped slashes and
# any non-ASCII character (em dashes appear inside character classes), so the
# body pattern must not exclude non-ASCII and must allow a bare / only when
# it is not escaped and the literal is properly terminated.
_JS_REGEX = re.compile(r"/((?:[^/\\\n]|\\.)+)/([gimsuy]*)")


def parse_regex_array(src: str, name: str) -> list[str]:
    blk = strip_js_comments(const_block(src, name))
    return ["/" + m.group(1) + "/" + m.group(2) for m in _JS_REGEX.finditer(blk)]


def parse_ai_phrases(src: str) -> list[dict]:
    blk = strip_js_comments(const_block(src, "AI_PHRASES"))
    out = []
    for m in re.finditer(
        r"\{\s*pattern:\s*/((?:[^/\\\n]|\\.)+)/([gimsuy]*)\s*,\s*tier:\s*(\d)\s*,\s*fix:\s*'((?:[^'\\]|\\.)*)'\s*\}",
        blk, re.S,
    ):
        out.append({"regex": "/" + m.group(1) + "/" + m.group(2),
                    "tier": int(m.group(3)), "fix": m.group(4)})
    return out


# --------------------------------------------------------------- patterns
# Each detector: id, name, category, weight, and the regex literals it uses.
# The regexes are pulled from the JS body of that detector so nothing is
# retyped by hand.
DETECTOR_RE = re.compile(
    r"\{\s*id:\s*(\d+),\s*name:\s*'([^']+)',\s*category:\s*'([^']+)',"
    r"\s*description:\s*'((?:[^'\\]|\\.)*)',"
    r"\s*weight:\s*(\d+),\s*detect\(text\)\s*\{",
    re.S,
)


_SHARED_CONSTS = (
    "SIGNIFICANCE_PHRASES",
    "PROMOTIONAL_WORDS",
    "VAGUE_ATTRIBUTION_PHRASES",
    "CHALLENGES_PHRASES",
    "COPULA_AVOIDANCE",
    "PRODUCT_POSITIONING_FOG",
    "HIDDEN_UNICODE_CHARS",
    "NON_BREAKING_SPACES",
)


def parse_detectors(src: str) -> list[dict]:
    src_nc = strip_js_comments(src)
    marks = list(DETECTOR_RE.finditer(src_nc))
    out = []
    for idx, m in enumerate(marks):
        start = m.end()
        if idx + 1 < len(marks):
            end = marks[idx + 1].start()
        else:
            # The final detector's body would otherwise run to EOF and
            # swallow the module's export/registry code.
            end = src_nc.find("module.exports", start)
            end = end if end > start else len(src_nc)
        body = src_nc[start:end]
        # Some detectors iterate a shared const array (SIGNIFICANCE_PHRASES,
        # PROMOTIONAL_WORDS, ...) instead of inlining their regexes. Record
        # WHICH consts a detector uses so the Python engine can look the
        # patterns up by name. The const bodies are NOT appended here: doing
        # so would make the scanner re-parse them and double-count.
        shared_consts = [
            name for name in _SHARED_CONSTS if re.search(rf"\b{name}\b", body)
        ]
        # Only regexes passed to the matcher helpers or .match()/.matchAll()
        # count. Scanning the whole body would pick up JS division operators
        # such as "words / 100" as if they were regex literals.
        # A regex literal in these sources appears as:
        #   const name = /re/gi;          (assignment, possibly across lines)
        #   findMatches(text, /re/gi, ...) (inline argument)
        #   text.match(/re/g)
        # Division operators look similar but are always followed by a
        # number or identifier rather than by a flag set and a delimiter.
        regexes = []
        for rx_match in _JS_REGEX.finditer(body):
            literal = "/" + rx_match.group(1) + "/" + rx_match.group(2)
            tail = body[rx_match.end():rx_match.end() + 80]
            head = body[max(0, rx_match.start() - 120):rx_match.start()]
            # Normalise whitespace so a regex split across lines is seen whole.
            flat_tail = " ".join(tail.split())
            flat_head = " ".join(head.split())
            assigned = bool(re.match(r"(?:const|let|var)\s+\w+\s*=\s*$", flat_head))
            inline_arg = bool(re.match(r"^[,)]", flat_tail))
            match_call = ".match(" in flat_head or ".matchAll(" in flat_head
            test_call = bool(re.match(r"^\.(?:test|exec)\(", flat_tail))
            # A division operator is followed by an operand, never by a
            # flag-terminated literal followed by ; or , or )
            looks_like_division = bool(re.match(r"^\s*[\w.)]", flat_tail)) and not (
                assigned or inline_arg or match_call or test_call
            )
            if not (assigned or inline_arg or match_call or test_call):
                continue
            if looks_like_division:
                continue
            # Final guard: a genuine regex literal always compiles. A
            # division operator misread as a literal does not, so this
            # rejects the false positives the context heuristics miss.
            # The check runs on the TRANSLATED pattern, because JS writes
            # astral code points as \u{XXXXXX} which Python rejects.
            try:
                re.compile(translate_js_regex(rx_match.group(1)))
            except re.error:
                continue
            regexes.append(literal)
        # The suggestions the reference implementation offers
        suggestions = re.findall(r"'((?:[^'\\]|\\.)*)'\s*,", body)
        out.append({
            "id": int(m.group(1)),
            "name": m.group(2),
            "category": m.group(3),
            "description": m.group(4),
            "weight": int(m.group(5)),
            "regexes": regexes,
            "shared_consts": shared_consts,
            "uses_ai_phrases": "AI_PHRASES" in body,
            "suggestions": suggestions,
        })
    return out


def main() -> int:
    vocab_src = read(SRC / "src" / "vocabulary.js")
    patterns_src = read(SRC / "src" / "patterns.js")

    data = {
        "tier1": parse_string_array(vocab_src, "TIER_1"),
        "tier2": parse_string_array(vocab_src, "TIER_2"),
        "tier3": parse_string_array(vocab_src, "TIER_3"),
        "ai_phrases": parse_ai_phrases(vocab_src),
        "detectors": parse_detectors(patterns_src),
        "significance": parse_regex_array(patterns_src, "SIGNIFICANCE_PHRASES"),
        "promotional": parse_regex_array(patterns_src, "PROMOTIONAL_WORDS"),
        "vague_attr": parse_regex_array(patterns_src, "VAGUE_ATTRIBUTION_PHRASES"),
        "challenges": parse_regex_array(patterns_src, "CHALLENGES_PHRASES"),
        "copula": parse_regex_array(patterns_src, "COPULA_AVOIDANCE"),
        "positioning": parse_regex_array(patterns_src, "PRODUCT_POSITIONING_FOG"),
    }

    if not data["detectors"]:
        print("ERROR: no detectors parsed")
        return 1
    if len(data["ai_phrases"]) < 50:
        print(f"ERROR: only {len(data['ai_phrases'])} AI phrases parsed")
        return 1

    out = OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")

    print(f"wrote {out}")
    print(f"  tier1={len(data['tier1'])} tier2={len(data['tier2'])} tier3={len(data['tier3'])}")
    print(f"  ai_phrases={len(data['ai_phrases'])} detectors={len(data['detectors'])}")
    total_rx = sum(len(d["regexes"]) for d in data["detectors"])
    print(f"  detector regexes={total_rx}")
    return 0


def translate_js_regex(body: str) -> str:
    """Convert a JS regex body to a Python-compatible one.

    JS writes astral code points with a braced escape, which Python's `re`
    module rejects; it wants the fixed-width capital-U form instead.
    """
    return re.sub(
        r"\\u\{([0-9A-Fa-f]{1,6})\}",
        lambda m: "\\U" + m.group(1).rjust(8, "0").upper(),
        body,
    )


def render(d: dict) -> str:
    def js_re(rx: str) -> str:
        """Convert a JS regex literal to a Python-compatible pattern string.

        The body runs from the opening "/" to the closing "/", skipping
        any slash that is escaped or that sits inside a character class.
        A naive rfind() would stop early on a character-class pattern,
        whose body is bounded by brackets rather than by a slash.
        """
        body = re.sub(r"/([gimsuy]+)$", "", rx)
        assert body.startswith("/")
        inner = body[1:]
        out = []
        in_class = False
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "\\" and i + 1 < len(inner):
                out.append(inner[i:i + 2])
                i += 2
                continue
            if ch == "[":
                in_class = True
            elif ch == "]":
                in_class = False
            elif ch == "/" and not in_class:
                break
            out.append(ch)
            i += 1
        return repr(translate_js_regex("".join(out)))

    lines = [
        '"""AI-writing signature catalogue.',
        "",
        "Generated by ``scripts/convert_reference_patterns.py`` from the reference",
        "implementations of the humanizer pattern engine. Do not hand-edit.",
        "",
        "Three vocabulary tiers of increasingly generic AI-flavored words, a set of",
        "regex-detected stock phrases, and 30 named detectors grouped by category.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
    ]

    lines.append("TIER_1: Final[tuple[str, ...]] = (")
    for w in d["tier1"]:
        lines.append(f"    {w!r},")
    lines.append(")")
    lines.append("")
    lines.append("TIER_2: Final[tuple[str, ...]] = (")
    for w in d["tier2"]:
        lines.append(f"    {w!r},")
    lines.append(")")
    lines.append("")
    lines.append("TIER_3: Final[tuple[str, ...]] = (")
    for w in d["tier3"]:
        lines.append(f"    {w!r},")
    lines.append(")")
    lines.append("")

    lines.append("# (regex, tier, fix hint)")
    lines.append("AI_PHRASES: Final[tuple[tuple[str, int, str], ...]] = (")
    for e in d["ai_phrases"]:
        lines.append(f"    ({js_re(e['regex'])!r}, {e['tier']}, {e['fix']!r}),")
    lines.append(")")
    lines.append("")

    for key, name in [
        ("significance", "SIGNIFICANCE_PHRASES"),
        ("promotional", "PROMOTIONAL_WORDS"),
        ("vague_attr", "VAGUE_ATTRIBUTION_PHRASES"),
        ("challenges", "CHALLENGES_PHRASES"),
        ("copula", "COPULA_AVOIDANCE"),
        ("positioning", "PRODUCT_POSITIONING_FOG"),
    ]:
        lines.append(f"{name}: Final[tuple[str, ...]] = (")
        for rx in d[key]:
            lines.append(f"    {js_re(rx)!r},")
        lines.append(")")
        lines.append("")

    lines.append("# id, name, category, weight, description, regexes")
    lines.append("PATTERNS: Final[tuple[dict, ...]] = (")
    for det in d["detectors"]:
        lines.append("    {")
        lines.append(f"        'id': {det['id']},")
        lines.append(f"        'name': {det['name']!r},")
        lines.append(f"        'category': {det['category']!r},")
        lines.append(f"        'weight': {det['weight']},")
        lines.append(f"        'description': {det['description']!r},")
        lines.append("        'regexes': (")
        for rx in det["regexes"]:
            lines.append(f"            {js_re(rx)!r},")
        lines.append("        ),")
        lines.append(f"        'uses_ai_phrases': {bool(det.get('uses_ai_phrases'))},")
        lines.append(f"        'shared_consts': {tuple(det.get('shared_consts', ()))!r},")
        lines.append("    },")
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
