"""Tests for the humanizer package.

The suite runs without an API key: the LLM layer is exercised through a
stub backend, and everything else is pure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from humanizer import analyze, data, preservation, profiles, structure
from humanizer.analyze import analyze as analyze_text
from humanizer.llm import EchoBackend, MiniMaxClient, MiniMaxError
from humanizer.pipeline import Humanizer
from humanizer.prompts import build_rewrite_prompt, build_review_prompt
from humanizer.structure import word_count

# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


AI_TEXT = """
It is important to note that the intricate tapestry of modern education
delves into a vibrant landscape of learning. This comprehensive framework
underscores the pivotal role of fostering meaningful engagement, showcasing
a testament to the transformative power of collaboration. Furthermore, the
robust ecosystem navigates the complexities of student development, offering
a seamless journey toward academic excellence. Moreover, this holistic
approach leverages cutting-edge methodologies, highlighting the profound
importance of continuous improvement.
"""

HUMAN_TEXT = """
We tried three times to get the thing running. The first attempt died after
twenty minutes, which we later traced to a missing config file. Second time
it crashed on startup, again something stupid. By the third attempt we had
read the logs properly. It worked. The fix was one line, of course.
"""


@pytest.fixture
def config():
    from humanizer.config import Config

    cfg = Config()
    cfg.max_passes = 2
    cfg.chunk_chars = 4000
    return cfg


class ScriptedBackend:
    """A backend that returns canned responses and records the calls."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, str]] = []

    def chat(self, user: str, system: str = "", temperature=None, max_tokens=None) -> str:
        self.calls.append((system, user))
        if self.responses:
            return self.responses.pop(0)
        return HUMAN_TEXT

    def list_models(self) -> list[str]:
        return ["scripted"]

    def check(self) -> tuple[bool, str]:
        return True, "scripted"


# ----------------------------------------------------------------------
# data catalogue
# ----------------------------------------------------------------------


class TestDataCatalogue:
    def test_three_vocabulary_tiers_are_populated(self):
        assert len(data.TIER_1) >= 50
        assert len(data.TIER_2) >= 100
        assert len(data.TIER_3) >= 50

    def test_tiers_do_not_overlap(self):
        t1, t2, t3 = set(data.TIER_1), set(data.TIER_2), set(data.TIER_3)
        assert not t1 & t2
        assert not t2 & t3

    def test_known_ai_terms_are_present(self):
        for word in ("delve", "tapestry", "robust", "showcase", "game-changer"):
            assert word in data.TIER_1

    def test_thirty_detectors_present(self):
        assert len(data.PATTERNS) == 30

    def test_detector_ids_are_unique_and_ordered(self):
        ids = [p["id"] for p in data.PATTERNS]
        assert ids == sorted(ids)
        assert len(set(ids)) == 30

    def test_every_detector_has_a_weight_and_category(self):
        for p in data.PATTERNS:
            assert 1 <= p["weight"] <= 5
            assert p["category"] in {
                "content", "language", "style", "communication", "filler",
            }

    def test_all_regexes_compile(self):
        import re

        for p in data.PATTERNS:
            for rx in p["regexes"]:
                re.compile(rx)
        for rx, _tier, _fix in data.AI_PHRASES:
            re.compile(rx)

    def test_ai_phrases_are_well_formed(self):
        assert len(data.AI_PHRASES) >= 50
        for rx, tier, fix in data.AI_PHRASES:
            assert 1 <= tier <= 3
            assert fix


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------


class TestAnalysis:
    def test_ai_text_scores_higher_than_human_text(self):
        ai = analyze_text(AI_TEXT)
        human = analyze_text(HUMAN_TEXT)
        assert ai.overall_score > human.overall_score

    def test_ai_text_triggers_multiple_detectors(self):
        report = analyze_text(AI_TEXT)
        detectors = {f.detector for f in report.findings}
        assert len(detectors) >= 4
        assert "AI vocabulary" in detectors

    def test_human_text_triggers_few_patterns(self):
        report = analyze_text(HUMAN_TEXT)
        assert len(report.findings) <= 2

    def test_delve_is_detected(self):
        report = analyze_text("We delve into the subject matter at length.")
        assert any(f.match.lower() == "delve" for f in report.findings)

    def test_furthermore_is_detected(self):
        report = analyze_text("The result was clear. Furthermore, it was fast.")
        assert any("furthermore" in f.match.lower() for f in report.findings)

    def test_curly_quotes_are_detected(self):
        report = analyze_text("She said \u201chello\u201d and left.")
        assert any("Curly quotes" == f.detector for f in report.findings)

    def test_em_dash_overuse_is_detected(self):
        text = "One thing \u2014 and another \u2014 and a third \u2014 happened."
        report = analyze_text(text)
        assert any("Em dash" in f.detector for f in report.findings)

    def test_burstiness_is_higher_for_human_text(self):
        human = analyze_text(HUMAN_TEXT).stats
        ai = analyze_text(AI_TEXT).stats
        assert human.burstiness > ai.burstiness

    def test_stats_on_empty_text(self):
        stats = analyze_text("").stats
        assert stats.word_count == 0
        assert stats.burstiness == 0.0

    def test_report_serialises(self):
        payload = analyze_text(AI_TEXT).as_dict()
        json.dumps(payload)
        assert payload["finding_count"] > 0
        assert "stats" in payload
        assert "detector_counts" in payload

    def test_findings_carry_line_numbers(self):
        report = analyze_text("First line.\nSecond line delves into it.")
        assert all(f.line >= 1 for f in report.findings)

    def test_score_band_is_reported(self):
        report = analyze_text(AI_TEXT)
        assert report.band in {
            "low - reads naturally",
            "moderate - some AI tells",
            "high - strong AI signature",
            "very high - pervasive AI signature",
        }


# ----------------------------------------------------------------------
# preservation
# ----------------------------------------------------------------------


class TestPreservation:
    def test_numbers_are_protected(self):
        text = "The study ran for 12 weeks with 340 participants."
        protected = preservation.extract(text)
        assert "12" not in protected.text
        assert "340" not in protected.text

    def test_urls_are_protected(self):
        text = "See https://example.com/paper for details."
        protected = preservation.extract(text)
        assert "example.com" not in protected.text

    def test_code_blocks_are_protected(self):
        text = "Run this:\n```python\nprint('hi')\n```"
        protected = preservation.extract(text)
        assert "print" not in protected.text

    def test_citations_are_protected(self):
        text = "As shown earlier [12], and again (Smith, 2020)."
        protected = preservation.extract(text)
        assert "[12]" not in protected.text

    def test_restore_returns_the_original(self):
        text = "The rate was 45% across 200 sites, see [3] and https://x.io."
        protected = preservation.extract(text)
        assert protected.restore(protected.text) == text

    def test_missing_detects_a_dropped_number(self):
        text = "We had 42 items and 7 errors."
        protected = preservation.extract(text)
        assert protected.missing("We had some items and 7 errors.") == ["42"]

    def test_strip_placeholders_removes_leftovers(self):
        assert preservation.strip_placeholders("text \u27e6P3\u27e7 more") == "text  more"

    def test_verify_helper(self):
        lost = preservation.verify("The answer is 42.", "The answer is unknown.")
        assert "42" in lost

    def test_placeholders_are_opaque(self):
        protected = preservation.extract("A 100 kg payload.")
        assert "\u27e6P" in protected.text


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------


class TestStructure:
    def test_split_blocks_separates_list_items(self):
        text = "Intro paragraph.\n\n- first item\n- second item\n\nOutro."
        blocks = structure.split_blocks(text)
        kinds = [b.kind for b in blocks]
        assert "list" in kinds
        assert kinds.count("list") == 2

    def test_list_markers_are_preserved(self):
        text = "- alpha\n- beta\n- gamma"
        blocks = structure.split_blocks(text)
        assert [b.prefix.strip() for b in blocks] == ["-", "-", "-"]

    def test_numbered_markers_are_preserved(self):
        text = "1. one\n2. two"
        blocks = structure.split_blocks(text)
        assert [b.prefix.strip() for b in blocks] == ["1.", "2."]

    def test_code_fence_stays_together(self):
        text = "Before\n\n```\ncode here\n```\n\nAfter"
        blocks = structure.split_blocks(text)
        assert any("code here" in b.body for b in blocks)

    def test_join_blocks_round_trips(self):
        text = "One paragraph.\n\nSecond paragraph."
        blocks = structure.split_blocks(text)
        assert structure.join_blocks(blocks) == text

    def test_restore_list_markers_fixes_dropped_numbers(self):
        original = "1. alpha\n2. beta"
        rewritten = "- alpha\n- beta"
        assert "1." in structure.restore_list_markers(original, rewritten)

    def test_word_count(self):
        assert word_count("one two three") == 3
        assert word_count("") == 0


# ----------------------------------------------------------------------
# profiles and prompts
# ----------------------------------------------------------------------


class TestProfilesAndPrompts:
    def test_five_profiles_defined(self):
        assert len(profiles.PROFILES) == 5

    def test_get_profile_falls_back(self):
        assert profiles.get_profile("nonexistent").key == "general"

    def test_profile_instruction_block_mentions_register(self):
        block = profiles.ACADEMIC.instruction_block()
        assert "register" in block.lower()
        assert len(block.splitlines()) >= 3

    def test_rewrite_prompt_contains_text_and_profile(self):
        system, user = build_rewrite_prompt("Some text.", profiles.ACADEMIC)
        assert "Some text." in user
        assert "academic" in system.lower() or "scholarly" in system.lower()

    def test_rewrite_prompt_includes_findings(self):
        system, _ = build_rewrite_prompt(
            "x", profiles.GENERAL, findings_summary='- "delve" (AI vocabulary)'
        )
        assert "delve" in system

    def test_review_prompt_asks_for_a_verdict(self):
        system, user = build_review_prompt("original", "rewritten")
        assert "VERDICT" in system
        assert "original" in user.lower()
        assert "rewritten" in user.lower()


# ----------------------------------------------------------------------
# LLM layer
# ----------------------------------------------------------------------


class TestLLM:
    def test_extract_text_anthropic_shape(self):
        payload = {"content": [{"type": "text", "text": "hello"}]}
        assert MiniMaxClient._extract_text(payload) == "hello"

    def test_extract_text_openai_shape(self):
        payload = {"choices": [{"message": {"content": "hello"}}]}
        assert MiniMaxClient._extract_text(payload) == "hello"

    def test_extract_text_ignores_thinking_blocks(self):
        payload = {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ]
        }
        assert MiniMaxClient._extract_text(payload) == "answer"

    def test_extract_text_returns_empty_on_garbage(self):
        assert MiniMaxClient._extract_text({"nope": 1}) == ""

    def test_client_requires_a_key(self):
        from humanizer.config import Config

        cfg = Config()
        cfg.api_key = ""
        with pytest.raises(MiniMaxError):
            MiniMaxClient(cfg)

    def test_anthropic_endpoint(self):
        from humanizer.config import Config

        cfg = Config()
        cfg.api_key = "test-key"
        cfg.api_style = "anthropic"
        client = MiniMaxClient(cfg)
        assert client._endpoint().endswith("/messages")

    def test_openai_endpoint(self):
        from humanizer.config import Config

        cfg = Config()
        cfg.api_key = "test-key"
        cfg.api_style = "openai"
        client = MiniMaxClient(cfg)
        assert client._endpoint().endswith("/chat/completions")

    def test_headers_carry_the_key(self):
        from humanizer.config import Config

        cfg = Config()
        cfg.api_key = "secret-key"
        cfg.api_style = "anthropic"
        client = MiniMaxClient(cfg)
        headers = client._headers()
        assert headers["Authorization"] == "Bearer secret-key"
        assert headers["x-api-key"] == "secret-key"

    def test_payload_shape_per_style(self):
        from humanizer.config import Config

        cfg = Config()
        cfg.api_key = "k"
        cfg.api_style = "anthropic"
        anthropic_payload = MiniMaxClient(cfg)._payload("sys", "usr", 0.5, 100)
        assert anthropic_payload["system"] == "sys"
        assert anthropic_payload["messages"][0]["role"] == "user"

        cfg.api_style = "openai"
        openai_payload = MiniMaxClient(cfg)._payload("sys", "usr", 0.5, 100)
        assert openai_payload["messages"][0]["role"] == "system"

    def test_echo_backend_returns_a_marker(self):
        out = EchoBackend().chat("hello")
        assert "dry-run" in out


# ----------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------


class TestPipeline:
    def test_rewrite_returns_a_result(self, config):
        backend = ScriptedBackend([HUMAN_TEXT])
        engine = Humanizer(backend, config)
        result = engine.rewrite(AI_TEXT, profile_key="general")
        assert result.original_report.overall_score > 0
        assert result.final_text

    def test_rewrite_reports_a_score_change(self, config):
        backend = ScriptedBackend([HUMAN_TEXT])
        engine = Humanizer(backend, config)
        result = engine.rewrite(AI_TEXT, profile_key="general")
        assert result.final_report.overall_score < result.original_report.overall_score

    def test_drift_is_detected_and_the_pass_is_reverted(self, config):
        backend = ScriptedBackend(
            [HUMAN_TEXT, "VERDICT: DRIFT\nDRIFTED_POINTS: the argument changed"]
        )
        engine = Humanizer(backend, config)
        result = engine.rewrite(AI_TEXT, profile_key="general", verify_meaning=True)
        assert result.passes
        assert result.passes[0].drift_detected
        assert any("drift" in n for n in result.notes)

    def test_a_clean_verdict_is_accepted(self, config):
        backend = ScriptedBackend([HUMAN_TEXT, "VERDICT: PASS"])
        engine = Humanizer(backend, config)
        result = engine.rewrite(AI_TEXT, profile_key="general", verify_meaning=True)
        assert not result.passes[0].drift_detected

    def test_a_failing_chunk_keeps_the_original(self, config):
        class Exploding(ScriptedBackend):
            def chat(self, user, system="", temperature=None, max_tokens=None):
                raise RuntimeError("boom")

        engine = Humanizer(Exploding(), config)
        result = engine.rewrite(AI_TEXT, profile_key="general")
        assert result.final_text  # still returns something usable

    def test_long_text_is_chunked(self, config):
        config.chunk_chars = 300
        backend = ScriptedBackend()
        engine = Humanizer(backend, config)
        long_text = "\n\n".join(f"Paragraph {i} discusses the topic." for i in range(12))
        engine.rewrite(long_text, profile_key="general", verify_meaning=False)
        assert len(backend.calls) > 1

    def test_result_serialises(self, config):
        backend = ScriptedBackend([HUMAN_TEXT])
        engine = Humanizer(backend, config)
        payload = engine.rewrite(AI_TEXT).as_dict()
        json.dumps(payload)
        assert "score_before" in payload

    def test_progress_callback_is_called(self, config):
        messages: list[str] = []
        backend = ScriptedBackend([HUMAN_TEXT])
        engine = Humanizer(backend, config, on_progress=messages.append)
        engine.rewrite(AI_TEXT)
        assert messages


# ----------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------


class TestCLI:
    def test_parser_builds(self):
        from humanizer.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "x.txt"])
        assert args.command == "analyze"

    def test_rewrite_accepts_a_profile(self):
        from humanizer.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["rewrite", "x.txt", "-p", "academic"])
        assert args.profile == "academic"

    def test_profiles_command_lists_profiles(self, capsys):
        from humanizer.cli import main

        main(["profiles"])
        captured = capsys.readouterr()
        assert "academic" in captured.out

    def test_analyze_command_works_on_a_file(self, tmp_path, capsys):
        from humanizer.cli import main

        path = tmp_path / "draft.md"
        path.write_text(AI_TEXT, encoding="utf-8")
        main(["analyze", str(path)])
        captured = capsys.readouterr()
        assert "Score" in captured.out
