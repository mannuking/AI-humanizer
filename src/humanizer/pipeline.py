"""The rewrite pipeline.

A single "rewrite this and hope" call is not enough. The tools in this
space that work do three things the naive version does not:

1. **Measure before and after.** The input is scored, the output is scored,
   and the score is what decides whether another pass is needed.
2. **Chunk.** Long documents are rewritten in pieces, because a model
   asked to rewrite ten thousand words at once will flatten them.
3. **Verify meaning.** A separate call checks that the rewrite did not
   drift. A rewrite that changes what the author said is a failure even if
   it scores beautifully.

The loop here is:

    analyse -> rewrite -> restore -> verify meaning -> re-score
                     ^                                |
                     +-------- not good enough --------+

Each pass gets the findings from the previous pass, so the second pass
knows what the first one failed to fix.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .analyze import Report, analyze
from .config import Config
from .preservation import extract, strip_placeholders
from .profiles import Profile, get_profile
from .prompts import build_review_prompt, build_rewrite_prompt
from .reasoning_guard import looks_like_reasoning
from .structure import (
    Block,
    join_blocks,
    restore_list_markers,
    split_blocks,
    word_count,
)
from .tokens import predictable_sentences
from .unicode_artifacts import normalize as normalize_artifacts


@dataclass
class PassResult:
    """What one rewrite pass produced."""

    text: str
    report: Report
    drift_detected: bool = False
    drift_notes: str = ""
    error: str = ""


@dataclass
class RewriteResult:
    """The outcome of the whole pipeline."""

    original: str
    final_text: str
    original_report: Report
    final_report: Report
    passes: list[PassResult] = field(default_factory=list)
    profile: Profile | None = None
    word_count_before: int = 0
    word_count_after: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """How much the score dropped (positive is good)."""
        return round(self.original_report.overall_score - self.final_report.overall_score, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_words": self.word_count_before,
            "final_words": self.word_count_after,
            "score_before": self.original_report.overall_score,
            "score_after": self.final_report.overall_score,
            "improvement": self.improvement,
            "passes": len(self.passes),
            "band_before": self.original_report.band,
            "band_after": self.final_report.band,
            "profile": self.profile.key if self.profile else None,
            "notes": self.notes,
        }


class Humanizer:
    """The rewrite engine."""

    def __init__(
        self,
        client,
        config: Config | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.config = config or Config()
        self.on_progress = on_progress or (lambda _msg: None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.on_progress(message)

    def _summarize_findings(self, report: Report, limit: int = 12) -> str:
        """A short list of the detected tells, for the prompt."""
        lines = []
        for f in report.findings[:limit]:
            lines.append(f'- "{f.match}" ({f.detector})')
        extra = len(report.findings) - limit
        if extra > 0:
            lines.append(f"- ...and {extra} more of the same kind")
        # Document-scale tells are structural, and the model needs to know
        # about them explicitly - a one-line closing paragraph cannot be
        # fixed by swapping a word.
        for f in report.document_findings[:6]:
            lines.append(f'- {f.detector}: "{f.match}" ({f.suggestion})')
        return "\n".join(lines)

    def _predictability_hint(self, text: str, top: int = 3) -> str:
        """Name the most predictable sentences, for the prompt.

        The commercial detectors measure how often the next most probable
        word is used. A sentence assembled entirely from common tokens is
        the sentence that reads as machine-written even when every word
        is individually ordinary, so those are the sentences worth
        rewriting first.
        """
        spans = predictable_sentences(text, top=top)
        if not spans:
            return ""
        lines = [
            "These sentences use only the most common vocabulary, which is "
            "the single strongest machine-writing signal. Rewrite them to "
            "use specific words:",
        ]
        for span in spans:
            lines.append(f'- "{span.text[:140]}"')
        return "\n".join(lines)

    def _chunk_blocks(self, blocks: list[Block]) -> list[list[Block]]:
        """Group blocks into chunks under the configured character budget.

        A reference block is never merged with prose: the bibliography
        travels as its own chunk and is passed through verbatim.
        """
        limit = self.config.chunk_chars
        chunks: list[list[Block]] = []
        current: list[Block] = []
        size = 0
        for block in blocks:
            block_size = len(block.text)
            if block.kind == "reference":
                if current:
                    chunks.append(current)
                    current = []
                    size = 0
                chunks.append([block])
                continue
            if current and size + block_size > limit:
                chunks.append(current)
                current = []
                size = 0
            current.append(block)
            size += block_size
        if current:
            chunks.append(current)
        return chunks

    def _rewrite_chunk(
        self,
        chunk: list[Block],
        profile: Profile,
        findings_summary: str,
        target_words: int | None,
        tone_instructions: str,
    ) -> tuple[str, list[str]]:
        """Rewrite one chunk of blocks.

        Returns the rewritten text and the list of protected values that
        did not survive. Protected spans (numbers, dates, citations, code,
        URLs) are hidden behind opaque placeholders during the call and
        restored afterwards, then checked - the incumbent tools lose
        between 6% and 35% of numeric tokens, and this is how we do not.
        """
        # A bibliography is not prose. Rewriting a citation changes the
        # reference itself, so it passes through untouched.
        if all(block.kind == "reference" for block in chunk):
            return join_blocks(chunk), []

        source = join_blocks(chunk)
        protected = extract(source)
        summary = findings_summary
        hint = self._predictability_hint(source)
        if hint:
            summary = f"{summary}\n\n{hint}".strip()
        system, user = build_rewrite_prompt(
            protected.text,
            profile,
            findings_summary=summary,
            target_words=target_words,
            tone_instructions=tone_instructions,
        )
        raw = self.client.chat(user, system=system)
        if looks_like_reasoning(raw):
            raise RuntimeError(
                "model returned internal reasoning instead of an answer"
            )
        restored = protected.restore(raw)
        # Any placeholder the model mangled or dropped leaves a stub.
        restored = strip_placeholders(restored)
        restored = restore_list_markers(source, restored)
        # Typographic artifacts - em dashes, curly quotes, non-standard
        # spaces - are a cheap signal for a detector and they break
        # copy-paste into a CMS or a database. Normalise them to plain
        # ASCII, which is what a person typing would produce.
        artifacts = normalize_artifacts(restored, smart_punctuation=False)
        restored = artifacts.text
        return restored.strip(), protected.missing(restored)

    def _verify_meaning(self, original: str, rewritten: str) -> tuple[bool, str]:
        """Ask the model whether the rewrite preserved the meaning.

        Returns ``(drift_detected, notes)``. A check that could not be
        completed counts as "no drift": an unavailable verifier must not
        discard a rewrite that may be fine.
        """
        try:
            system, user = build_review_prompt(original, rewritten)
            verdict = self.client.chat(user, system=system)
        except Exception as exc:  # noqa: BLE001 - a failed check is not a failed rewrite
            self._log(f"  meaning check skipped: {exc}")
            return False, ""

        match = re.search(r"VERDICT:\s*(PASS|DRIFT)", verdict, re.IGNORECASE)
        if not match:
            return False, ""
        if match.group(1).upper() == "PASS":
            return False, ""
        notes = ""
        note_match = re.search(r"DRIFTED_POINTS:\s*(.+)", verdict, re.IGNORECASE)
        if note_match:
            notes = note_match.group(1).strip()
        return True, notes

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------
    def rewrite(
        self,
        text: str,
        profile_key: str = "general",
        tone_instructions: str = "",
        max_passes: int | None = None,
        preserve_word_count: bool | None = None,
        verify_meaning: bool = True,
    ) -> RewriteResult:
        """Rewrite ``text`` and return the full result."""
        profile = get_profile(profile_key)
        max_passes = self.config.max_passes if max_passes is None else max_passes
        preserve_words = (
            self.config.preserve_word_count
            if preserve_word_count is None
            else preserve_word_count
        )

        original_report = analyze(text)
        before_words = word_count(text)
        notes: list[str] = []
        passes: list[PassResult] = []

        self._log(f"Analysed input: score {original_report.overall_score:.3f} "
                  f"({original_report.band}), {len(original_report.findings)} findings")

        # Clean the input before anything else: typographic artifacts in
        # the source would otherwise survive into the result.
        incoming = normalize_artifacts(text, smart_punctuation=False)
        if incoming.changed:
            self._log(f"  normalised {incoming.changed} typographic character(s)")
            text = incoming.text
            original_report = analyze(text)

        current_text = text
        current_report = original_report
        lost_spans: list[str] = []

        for pass_no in range(1, max(1, max_passes) + 1):
            self._log(f"Pass {pass_no}/{max_passes}: rewriting")

            blocks = split_blocks(current_text)
            chunks = self._chunk_blocks(blocks)
            summary = self._summarize_findings(current_report)
            target = before_words if preserve_words else None

            pieces: list[str] = []
            chunk_errors: list[str] = []
            lost_spans.clear()
            for idx, chunk in enumerate(chunks, 1):
                chunk_words = word_count(join_blocks(chunk))
                chunk_target = (
                    round(chunk_words * (before_words / max(before_words, 1)))
                    if preserve_words and before_words
                    else None
                )
                # Scale the chunk target by the chunk's share of the text.
                if preserve_words and before_words:
                    chunk_target = round(
                        chunk_words * (target / before_words) if target else chunk_words
                    )
                try:
                    rewritten_chunk, lost = self._rewrite_chunk(
                        chunk, profile, summary, chunk_target, tone_instructions
                    )
                    pieces.append(rewritten_chunk)
                    if lost:
                        lost_spans.extend(lost)
                        self._log(
                            f"  chunk {idx}: {len(lost)} protected value(s) lost"
                        )
                except Exception as exc:  # noqa: BLE001
                    chunk_errors.append(f"chunk {idx}: {exc}")
                    pieces.append(join_blocks(chunk))  # keep the original
                    self._log(f"  chunk {idx} failed, kept original: {exc}")

            rewritten = "\n\n".join(p for p in pieces if p.strip())
            if not rewritten.strip():
                rewritten = current_text
                notes.append(f"pass {pass_no}: rewrite produced nothing, kept input")

            new_report = analyze(rewritten)

            drift = False
            drift_notes = ""
            if verify_meaning and pass_no == 1:
                self._log("  checking that the meaning survived")
                drift, drift_notes = self._verify_meaning(current_text, rewritten)
                if drift:
                    notes.append(f"pass {pass_no}: meaning drift detected"
                                 + (f" - {drift_notes}" if drift_notes else ""))
                    self._log("  meaning drift detected, reverting this pass")

            result = PassResult(
                text=rewritten,
                report=new_report,
                drift_detected=drift,
                drift_notes=drift_notes,
                error="; ".join(chunk_errors),
            )
            passes.append(result)

            if drift:
                # Keep the previous, safer version.
                break

            current_text = rewritten
            current_report = new_report

            self._log(
                f"  score {original_report.overall_score:.3f} -> "
                f"{new_report.overall_score:.3f}"
            )

            if new_report.overall_score <= self.config.target_score:
                self._log(f"  reached target score {self.config.target_score:.2f}")
                break
            if pass_no >= max_passes:
                break
            # Only spend another pass if there is something left to fix.
            if not new_report.findings and new_report.style_score < 0.15:
                break

        after_words = word_count(current_text)
        if before_words:
            ratio = after_words / before_words
            # The independent benchmark gates a rewrite at 0.6-1.4x the
            # input length; the best tools sit near 1.05x. Walter Writes
            # averages 1.349x and loses nine penalty points for it.
            if ratio > 1.4:
                notes.append(
                    f"length inflated {ratio:.2f}x ({before_words} -> {after_words} words)"
                )
            elif ratio < 0.6:
                notes.append(
                    f"length deflated {ratio:.2f}x ({before_words} -> {after_words} words)"
                )
            elif preserve_words and abs(ratio - 1.0) > self.config.word_count_tolerance:
                notes.append(
                    f"word count moved {before_words} -> {after_words} "
                    f"({ratio:.2f}x)"
                )
        if lost_spans:
            notes.append(
                f"{len(lost_spans)} protected value(s) lost in the rewrite: "
                + ", ".join(sorted(set(lost_spans))[:8])
            )

        return RewriteResult(
            original=text,
            final_text=current_text,
            original_report=original_report,
            final_report=current_report,
            passes=passes,
            profile=profile,
            word_count_before=before_words,
            word_count_after=after_words,
            notes=notes,
        )
