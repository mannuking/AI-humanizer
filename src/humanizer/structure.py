"""Document structure handling.

A rewrite must not lose the author's formatting: bullet markers, numbered
lists, headings, code blocks, tables, citations and inline math all have to
come back out the other side intact.

The approach is to *protect* those regions with opaque placeholders before
the text goes to the model, then restore them afterwards. The model never
sees a code block or a citation, so it cannot mangle one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Protects fenced code blocks, then inline code.
_FENCED_CODE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
#: Protects LaTeX-ish maths.
_MATH = re.compile(r"(\$\$[^$]*\$\$|\\\[[^\]]*\\\]|\\\([^)]*\\\)|\$[^$\n]+\$)")
#: Protects markdown tables (the whole block, header through last row).
_TABLE = re.compile(r"(?:^\|.*\|[ \t]*$\n?)+", re.MULTILINE)
#: Protects URLs and DOIs - they must survive verbatim.
_URL = re.compile(r"\b(?:https?://|doi:|10\.\d{4,9}/)[^\s)\]]+", re.IGNORECASE)
#: Protects bracketed numeric citations: [1], [2, 3], (Smith, 2020).
_CITATION = re.compile(r"\[[\d\s,;–-]{1,40}\]|\([A-Z][A-Za-z'’\-]+(?: et al\.)?,?\s*\d{4}[a-z]?\)")

#: Order matters - the most protected region is captured first.
_PROTECTORS = (
    ("CODE_BLOCK", _FENCED_CODE),
    ("TABLE", _TABLE),
    ("MATH", _MATH),
    ("INLINE_CODE", _INLINE_CODE),
    ("URL", _URL),
    ("CITATION", _CITATION),
)


@dataclass
class ProtectedText:
    """Text with protected regions replaced by placeholders."""

    text: str
    regions: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        """Put the protected regions back."""
        out = text
        # Longest placeholder first, so [P0] cannot shadow [P10].
        for key in sorted(self.regions, key=lambda k: len(k), reverse=True):
            out = out.replace(key, self.regions[key])
        return out


def protect(text: str) -> ProtectedText:
    """Replace code, tables, maths, URLs and citations with placeholders."""
    regions: dict[str, str] = {}
    counter = 0
    working = text

    for label, pattern in _PROTECTORS:
        def _swap(match: re.Match, label=label) -> str:
            nonlocal counter
            token = f"[PH{counter}:{label}]"
            regions[token] = match.group(0)
            counter += 1
            return token

        working = pattern.sub(_swap, working)

    return ProtectedText(text=working, regions=regions)


# --------------------------------------------------------------------------
# structural decomposition
# --------------------------------------------------------------------------


@dataclass
class Block:
    """One rewrite unit."""

    kind: str
    prefix: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.prefix}{self.body}"


_LIST_ITEM = re.compile(r"^(\s*)(?:([-*•])|(\d+[.)])|([a-z][.)]))(\s+)(.*)$")
_HEADING = re.compile(r"^(#{1,6}\s+.*|\*\*[^*]+\*\*\s*:?|\d+\.\s+[A-Z][^\n:]{2,50}:)\s*$")


def split_blocks(text: str) -> list[Block]:
    """Split text into blocks that must each be rewritten on their own.

    A block is a paragraph, a list item, a heading, or a fenced code block.
    Keeping list items separate means the model cannot merge two bullets
    into one paragraph.
    """
    blocks: list[Block] = []
    lines = text.split("\n")
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if buffer:
            body = "\n".join(buffer).strip("\n")
            if body.strip():
                blocks.append(Block(kind="paragraph", prefix="", body=body))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        # Fenced code blocks are atomic.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = marker
                buffer.append(line)
                continue
            if marker == fence_marker:
                buffer.append(line)
                flush()
                in_fence = False
                fence_marker = ""
                continue
        if in_fence:
            buffer.append(line)
            continue

        if not stripped:
            flush()
            continue

        m = _LIST_ITEM.match(line)
        if m:
            flush()
            indent, bullet, number, alpha, space, body = m.groups()
            marker = bullet or number or alpha
            blocks.append(
                Block(kind="list", prefix=f"{indent}{marker}{space}", body=body)
            )
            continue

        if _HEADING.match(line):
            flush()
            blocks.append(Block(kind="heading", prefix="", body=line.strip()))
            continue

        buffer.append(line)

    flush()
    return blocks


def join_blocks(blocks: list[Block]) -> str:
    """Reassemble blocks into text, preserving blank-line separation."""
    parts: list[str] = []
    for block in blocks:
        if block.kind == "list":
            parts.append(block.text)
        else:
            parts.append(block.body)
    return "\n\n".join(parts)


def restore_list_markers(original: str, rewritten: str) -> str:
    """Put the original list markers back if the model dropped them.

    Models sometimes render a numbered list without the numbers. The
    rewritten text keeps the original marker sequence.
    """
    orig_items = [
        (_LIST_ITEM.match(line).group(1) or "",
         _LIST_ITEM.match(line).group(2) or "",
         _LIST_ITEM.match(line).group(3) or "",
         _LIST_ITEM.match(line).group(4) or "",
         _LIST_ITEM.match(line).group(5) or "")
        for line in original.split("\n")
        if _LIST_ITEM.match(line)
    ]
    if not orig_items:
        return rewritten

    out_lines = rewritten.split("\n")
    orig_idx = 0
    for i, line in enumerate(out_lines):
        m = _LIST_ITEM.match(line)
        if not m:
            continue
        if orig_idx >= len(orig_items):
            break
        indent, bullet, number, alpha, space = orig_items[orig_idx]
        orig_idx += 1
        if bullet:
            out_lines[i] = f"{indent}{bullet}{space}{m.group(6)}"
        elif number:
            out_lines[i] = f"{indent}{number}{space}{m.group(6)}"
        elif alpha:
            out_lines[i] = f"{indent}{alpha}{space}{m.group(6)}"
    return "\n".join(out_lines)


def word_count(text: str) -> int:
    """Count words the way the UI reports them."""
    return len(re.findall(r"\b[\w'-]+\b", text))
