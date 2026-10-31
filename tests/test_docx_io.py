"""Tests for the Word document reader and writer.

The behaviour that matters here is the round trip: a document read from
disk and written back must come out with the same text, the same
styles, and a file Word can open. The reference-section detection is
tested too, because a wrong answer there silently stops the humanizer
rewriting most of a document.
"""

from __future__ import annotations

import zipfile

import pytest

from humanizer import docx_io

docx = pytest.importorskip("docx")


def _make_document(path):
    """A small document with headings, a list and a reference section."""
    import docx as _docx

    document = _docx.Document()
    document.add_heading("Introduction", level=1)
    document.add_paragraph("The opening paragraph of the document.")
    document.add_paragraph("First item", style="List Number")
    document.add_paragraph("Second item", style="List Number")
    document.add_heading("Body", level=2)
    document.add_paragraph("A paragraph under the second heading.")
    document.add_heading("References", level=1)
    document.add_paragraph("Smith, J. (2020) A Title. London: A Publisher.")
    document.add_paragraph("Jones, A. (2021) Another Title. Delhi: Another Publisher.")
    document.save(str(path))
    return path


class TestRead:
    def test_reads_paragraphs_and_text(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        document = docx_io.read(path)
        assert document.paragraphs
        assert "Introduction" in document.text
        assert "A paragraph under the second heading." in document.text

    def test_summary_counts(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        summary = docx_io.summary(docx_io.read(path))
        assert summary["paragraphs"] > 0
        assert summary["words"] > 0
        assert summary["references"] == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            docx_io.read(tmp_path / "nothing.docx")

    def test_non_docx_raises_a_useful_error(self, tmp_path):
        path = tmp_path / "plain.docx"
        path.write_text("this is not a Word document", encoding="utf-8")
        with pytest.raises(ValueError, match=r"not a \.docx file"):
            docx_io.read(path)

    def test_legacy_doc_gives_conversion_instructions(self, tmp_path):
        path = tmp_path / "old.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0legacy compound file")
        with pytest.raises(ValueError, match=r"convert-to docx"):
            docx_io.read(path)


class TestReferenceDetection:
    def test_flags_only_the_reference_section(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        document = docx_io.read(path)
        flagged = [p.text for p in document.paragraphs if p.is_reference]
        assert len(flagged) == 2
        assert "Smith, J." in flagged[0]

    def test_a_bibliography_heading_ends_at_the_next_heading(self, tmp_path):
        """A mid-document heading called Bibliography is not the end."""
        import docx as _docx

        document = _docx.Document()
        document.add_heading("Bibliography", level=1)
        document.add_paragraph("A listed source.")
        document.add_heading("Appendix", level=1)
        document.add_paragraph("This paragraph is not a reference.")
        path = tmp_path / "mid.docx"
        document.save(str(path))

        result = docx_io.read(path)
        flagged = [p.text for p in result.paragraphs if p.is_reference]
        assert flagged == ["A listed source."]


class TestRoundTrip:
    def test_text_and_styles_survive(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        original = docx_io.read(path)
        out = tmp_path / "out.docx"
        docx_io.write(original, out)

        again = docx_io.read(out)
        assert again.text == original.text
        assert [p.style for p in again.paragraphs] == [
            p.style for p in original.paragraphs
        ]

    def test_output_is_a_real_zip_package(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        out = tmp_path / "out.docx"
        docx_io.write(docx_io.read(path), out)
        assert zipfile.is_zipfile(out)
        with zipfile.ZipFile(out) as package:
            names = package.namelist()
            assert "word/document.xml" in names
            assert "[Content_Types].xml" in names

    def test_a_fresh_document_can_be_written(self, tmp_path):
        """The writer works with no source document behind it."""
        out = tmp_path / "new.docx"
        document = docx_io.Document(
            paragraphs=docx_io.to_docx_paragraphs("First para.\n\nSecond para.")
        )
        docx_io.write(document, out)
        assert zipfile.is_zipfile(out)
        again = docx_io.read(out)
        assert "First para." in again.text
        assert "Second para." in again.text


class TestApplyText:
    def test_replaces_paragraphs_in_place(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        document = docx_io.read(path)
        rewritten = document.text.replace("opening", "closing")
        updated = docx_io.apply_text(document, rewritten)
        assert "closing paragraph" in updated.text
        assert "opening paragraph" not in updated.text

    def test_keeps_the_styles_of_the_original(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        document = docx_io.read(path)
        updated = docx_io.apply_text(document, document.text.upper())
        assert [p.style for p in updated.paragraphs] == [
            p.style for p in document.paragraphs
        ]

    def test_extra_text_is_kept(self, tmp_path):
        path = _make_document(tmp_path / "sample.docx")
        document = docx_io.read(path)
        updated = docx_io.apply_text(document, document.text + "\n\nAn extra paragraph.")
        assert "An extra paragraph." in updated.text
