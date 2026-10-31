"""Read and write Word documents.

The humanizer works on text. Word documents are what people actually
have, and a tool that only reads ``.txt`` forces the user to strip the
document down before they can use it - which loses the structure that
the preservation layer exists to protect.

This module handles the round trip:

  * text is extracted paragraph by paragraph, keeping the paragraph
    boundaries, the list markers, the headings and the reference
    section
  * the rewrite is written back into the same document, preserving the
    run formatting of the paragraph it replaces where the text is
    unchanged, and preserving the styles of headings and lists

It uses ``python-docx`` when it is installed and falls back to reading
the WordprocessingML directly, so the core tool has no hard dependency
on it.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


#: A Word paragraph, as text.
@dataclass
class Paragraph:
    """One paragraph of a Word document."""

    text: str
    style: str = ""
    index: int = 0
    #: True when the paragraph is part of a list, by style or numbering.
    is_list: bool = False
    #: True when the paragraph looks like a bibliography entry.
    is_reference: bool = False


@dataclass
class Document:
    """A Word document, as text plus enough structure to write it back."""

    paragraphs: list[Paragraph] = field(default_factory=list)
    #: The original file path, when there was one.
    path: Path | None = None

    @property
    def text(self) -> str:
        """The whole document as plain text."""
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def to_text(self) -> str:
        """The document as text, ready to hand to the analyzer."""
        return self.text


def _docx_namespace() -> dict[str, str]:
    return {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }


def _read_with_python_docx(path: Path) -> list[Paragraph]:
    """Read via python-docx when it is available."""
    import docx  # type: ignore[import-not-found]

    document = docx.Document(str(path))
    paragraphs: list[Paragraph] = []
    for para in document.paragraphs:
        style = (para.style.name if para.style is not None else "") or ""
        paragraphs.append(
            Paragraph(
                text=para.text,
                style=style,
                index=len(paragraphs),
                is_list=style.lower().startswith("list")
                or para._p.find(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr"
                )
                is not None,
            )
        )
    return paragraphs


def _read_with_xml(path: Path) -> list[Paragraph]:
    """Read the WordprocessingML directly.

    This is the fallback when python-docx is not installed. It handles
    paragraphs, their style and their list numbering, which is enough
    for the humanizer's needs.
    """
    import xml.etree.ElementTree as ET

    ns = _docx_namespace()
    w = f"{{{ns['w']}}}"

    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")

    root = ET.fromstring(xml)
    body = root.find(f"{w}body")
    if body is None:
        return []

    paragraphs: list[Paragraph] = []
    for i, p in enumerate(body.iter(f"{w}p")):
        parts: list[str] = []
        for node in p.iter():
            if node.tag == f"{w}t":
                parts.append(node.text or "")
            elif node.tag == f"{w}tab":
                parts.append("\t")
            elif node.tag in (f"{w}br", f"{w}cr"):
                parts.append("\n")
        text = "".join(parts)

        style = ""
        ppr = p.find(f"{w}pPr")
        if ppr is not None:
            pstyle = ppr.find(f"{w}pStyle")
            if pstyle is not None:
                style = pstyle.get(f"{w}val", "") or ""

        is_list = style.lower().startswith("list")
        if ppr is not None and ppr.find(f"{w}numPr") is not None:
            is_list = True

        paragraphs.append(
            Paragraph(
                text=text,
                style=style,
                index=i,
                is_list=is_list,
            )
        )
    return paragraphs


#: The headings that start a bibliography.
_REFERENCE_HEADING = re.compile(
    r"^\s*(references|bibliography|works\s+cited|literature\s+cited"
    r"|notes\s+and\s+references)\s*:?\s*$",
    re.IGNORECASE,
)


def _heading_level(style: str) -> int:
    """The outline level of a paragraph, or 0 when it is not a heading."""
    match = re.match(r"heading\s*(\d+)", style.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _mark_references(paragraphs: list[Paragraph]) -> None:
    """Flag the paragraphs that belong to a bibliography.

    The humanizer never rewrites a reference section - rewriting a
    citation falsifies the attribution - so it has to know where that
    section starts.

    A reference section runs from its heading to the next heading of
    the same level or higher. Without that stopping point, a heading
    called "Bibliography" part-way through a document would mark
    everything after it as a reference, and the rewrite would leave
    most of the document untouched.
    """
    in_references = False
    level = 0

    for para in paragraphs:
        text = para.text.strip()
        current_level = _heading_level(para.style)

        if current_level:
            # A new heading ends the reference section, unless it is
            # itself the heading that started one.
            if in_references and current_level <= level:
                in_references = False
            if _REFERENCE_HEADING.match(text):
                in_references = True
                level = current_level
                continue

        if in_references and text:
            para.is_reference = True


def read(path: str | Path) -> Document:
    """Read a Word document.

    Raises ``ValueError`` for a file that is not a Word document, and
    ``FileNotFoundError`` when it does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"No such file: {target}")

    # A .docx is a zip. If it is not, it is a legacy .doc and needs a
    # conversion first.
    if not zipfile.is_zipfile(target):
        raise ValueError(
            f"{target.name} is not a .docx file. Legacy .doc files must be "
            f"converted first: soffice --headless --convert-to docx {target.name}"
        )

    try:
        paragraphs = _read_with_python_docx(target)
    except ImportError:
        paragraphs = _read_with_xml(target)
    except Exception:  # noqa: BLE001 - fall back rather than fail
        paragraphs = _read_with_xml(target)

    _mark_references(paragraphs)
    return Document(paragraphs=paragraphs, path=target)


def write(document: Document, path: str | Path) -> Path:
    """Write a document back out, keeping the styles.

    The paragraphs are matched to the originals by position, and the
    text of each is replaced. Styles, numbering and the overall
    document structure are left alone - this rewrites prose, it does
    not rebuild the file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import docx  # type: ignore[import-not-found]
    except ImportError:
        return _write_with_xml(document, target)

    out = docx.Document(str(document.path) if document.path else None)
    if len(out.paragraphs) != len(document.paragraphs):
        # The paragraph count changed, which python-docx cannot do
        # safely. Write a fresh valid document rather than risking a
        # corrupt file.
        return _write_with_xml(document, target)

    for para, new in zip(out.paragraphs, document.paragraphs, strict=False):
        if new.text == para.text:
            continue
        runs = para.runs
        if not runs:
            para.text = new.text
            continue
        # Put the whole new text in the first run, keeping that run's
        # formatting, and clear the others.
        runs[0].text = new.text
        for run in runs[1:]:
            run.text = ""

    out.save(str(target))
    return target


#: The parts a minimal Word document needs. Without these Word
#: refuses to open the file.
_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _write_with_xml(document: Document, target: Path) -> Path:
    """Write a valid .docx when python-docx is not installed.

    Writing plain text with a .docx extension produces a file Word
    cannot open, so the fallback builds a real package - the minimum
    set of parts a Word document needs, with one paragraph per
    document paragraph.
    """
    import xml.etree.ElementTree as ET

    ns = _docx_namespace()
    w = f"{{{ns['w']}}}"
    ET.register_namespace("w", ns["w"])

    body = ET.Element(f"{w}body")
    for para in document.paragraphs:
        p = ET.SubElement(body, f"{w}p")
        if para.text:
            run = ET.SubElement(p, f"{w}r")
            t = ET.SubElement(run, f"{w}t")
            # xml:space preserves leading and trailing spaces.
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            t.text = para.text
    # A section is required for the document to be valid.
    ET.SubElement(body, f"{w}sectPr")

    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns["w"]}">'
        f'<w:body>{"".join(ET.tostring(c, encoding="unicode") for c in body)}</w:body>'
        "</w:document>"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("word/document.xml", xml)
    return target


def to_docx_paragraphs(text: str) -> list[Paragraph]:
    """Split plain text into paragraphs.

    Used when the humanizer's output has to go back into a document
    that was not read from one.
    """
    out: list[Paragraph] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        out.append(Paragraph(text=block, index=len(out)))
    return out


def apply_text(document: Document, new_text: str) -> Document:
    """Put rewritten text back into a document, paragraph by paragraph.

    The rewrite is matched to the original paragraphs in order. Where
    the counts differ - the rewrite split or merged a paragraph - the
    extra text is appended to the last paragraph rather than being
    dropped.
    """
    new_paragraphs = to_docx_paragraphs(new_text)
    if not new_paragraphs:
        return document

    out: list[Paragraph] = []
    for i, original in enumerate(document.paragraphs):
        if i < len(new_paragraphs):
            para = Paragraph(
                text=new_paragraphs[i].text,
                style=original.style,
                index=i,
                is_list=original.is_list,
                is_reference=original.is_reference,
            )
        else:
            para = Paragraph(
                text=original.text,
                style=original.style,
                index=i,
                is_list=original.is_list,
                is_reference=original.is_reference,
            )
        out.append(para)

    # Any surplus from the rewrite goes to the end.
    if len(new_paragraphs) > len(document.paragraphs):
        surplus = "\n".join(p.text for p in new_paragraphs[len(document.paragraphs):])
        if out:
            out[-1].text = f"{out[-1].text}\n{surplus}".strip()
        else:
            out = to_docx_paragraphs(surplus)

    return Document(paragraphs=out, path=document.path)


def summary(document: Document) -> dict[str, Any]:
    """A small description of a document, for the CLI and the UI."""
    return {
        "path": str(document.path) if document.path else None,
        "paragraphs": len(document.paragraphs),
        "words": document.words,
        "references": sum(1 for p in document.paragraphs if p.is_reference),
        "lists": sum(1 for p in document.paragraphs if p.is_list),
    }
