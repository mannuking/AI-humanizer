"""Command-line interface.

``humanize`` with no arguments starts the web UI. Everything else is a
subcommand:

    humanize                     launch the Streamlit UI
    humanize rewrite FILE        rewrite a file
    humanize analyze FILE        score a file, no API calls
    humanize profiles            list the writing profiles
    humanize check               verify the API key and model
    humanize models              list the models the key can use
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from humanizer import perplexity as px
from humanizer.analyze import analyze

from .config import load_config
from .llm import MiniMaxError, build_client
from .pipeline import Humanizer
from .profiles import describe_profiles, profile_keys

PACKAGE_ROOT = Path(__file__).resolve().parent
WEB_APP = PACKAGE_ROOT / "web" / "app.py"


def _read_input(path: str | None) -> str:
    """Read the input, whatever it is.

    Word documents are read paragraph by paragraph so that the
    structure - headings, lists, the bibliography - survives.
    """
    if path in (None, "-"):
        return sys.stdin.read()
    target = Path(path)
    if target.suffix.lower() == ".docx":
        from . import docx_io

        return docx_io.read(target).to_text()
    if target.suffix.lower() == ".doc":
        raise ValueError(
            f"{target.name} is a legacy .doc file. Convert it first:\n"
            f"  soffice --headless --convert-to docx {target.name}"
        )
    return target.read_text(encoding="utf-8")


def _write_output(path: str | None, text: str, source: str | None) -> None:
    """Write the result, keeping the input format.

    Rewriting a .docx and saving it as .txt would throw away the
    document. When the input was a Word file, the output is a Word file.
    """
    if not path:
        print(text)
        return
    target = Path(path)
    if target.suffix.lower() == ".docx":
        from . import docx_io

        if source:
            document = docx_io.read(source)
            document = docx_io.apply_text(document, text)
            docx_io.write(document, target)
        else:
            document = docx_io.Document(paragraphs=docx_io.to_docx_paragraphs(text))
            docx_io.write(document, target)
        print(f"Wrote {target}")
        return
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}")


def _print_report(text: str, verbose: bool = True) -> None:
    # Measure against a local model when one is available - that is the
    # measurement the commercial detectors make, and it is what catches
    # technical prose the phrase catalogue misses.
    report = analyze(text, with_perplexity=px.available())
    stats = report.stats
    print(f"\n  Score   {report.overall_score:.3f}  ({report.band})")
    print(f"          pattern {report.pattern_score:.3f}   style {report.style_score:.3f}")
    print(
        f"  Words   {stats.word_count}   sentences {stats.sentence_count}   "
        f"paragraphs {stats.paragraph_count}"
    )
    print(
        f"  Shape   burstiness {stats.burstiness:.3f}   "
        f"TTR {stats.type_token_ratio:.3f}   "
        f"reading ease {stats.reading_ease:.1f}"
    )
    print(
        f"  Layers  lexical {len(report.findings)}   "
        f"threshold {len(report.counted_findings)}   "
        f"document {len(report.document_findings)}"
    )
    tp = report.token_profile
    print(
        f"  Tokens  {tp.token_count}   "
        f"generic share {tp.generic_token_share:.3f}   "
        f"hapax {tp.hapax_token_ratio:.3f}   "
        f"chars/token {tp.chars_per_token:.2f}"
    )
    if report.perplexity is not None and not report.perplexity.unavailable:
        burst = (
            f"   burstiness {report.burstiness.cv:.3f}"
            if report.burstiness
            else ""
        )
        print(
            f"  Model   perplexity {report.perplexity.perplexity:.2f}   "
            f"nll {report.perplexity.mean_nll:.4f}   "
            f"tokens {report.perplexity.token_count}{burst}"
        )
    if report.artifacts.found:
        print(f"  Typographic artifacts ({len(report.artifacts.found)} kinds):")
        for name, count in report.artifacts.found.items():
            print(f"    {name}: {count}")
    counted = report.counted_findings
    docs = report.document_findings
    total = len(report.findings) + len(counted) + len(docs)
    if not total:
        print("\n  No AI-writing patterns detected.")
        return
    print(f"\n  Findings ({total}):")
    for finding in report.top_findings(15 if verbose else 5):
        print(
            f"    line {finding.line:>3}  [{finding.category}] "
            f"{finding.detector}: \"{finding.match[:52]}\""
        )
    for finding in counted[:10]:
        print(f"    line {finding.line:>3}  [{finding.category}] "
              f"{finding.detector}: {finding.match[:52]}")
    for finding in docs[:10]:
        print(f"    line {finding.line:>3}  [structure] "
              f"{finding.detector}: {finding.match[:52]}")
    counts = report.detector_counts()
    if verbose and counts:
        print("\n  By detector:")
        for name, count in counts.items():
            print(f"    {count:>3}  {name}")


def cmd_analyze(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    if not text.strip():
        print("Nothing to analyse.", file=sys.stderr)
        return 1
    _print_report(text, verbose=not args.quiet)
    return 0


def cmd_rewrite(args: argparse.Namespace) -> int:
    config = load_config()
    text = _read_input(args.file)
    if not text.strip():
        print("Nothing to rewrite.", file=sys.stderr)
        return 1

    try:
        client = build_client(config, dry_run=args.dry_run)
    except MiniMaxError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    engine = Humanizer(client, config, on_progress=progress)
    result = engine.rewrite(
        text,
        profile_key=args.profile,
        tone_instructions=args.instructions or "",
        max_passes=args.passes,
        preserve_word_count=not args.no_preserve_length,
        verify_meaning=not args.no_verify,
    )

    if args.output:
        try:
            _write_output(args.output, result.final_text, args.file)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        print(result.final_text)

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"  {result.original_report.overall_score:.3f} -> "
          f"{result.final_report.overall_score:.3f}  "
          f"(improvement {result.improvement:+.3f})", file=sys.stderr)
    print(f"  words {result.word_count_before} -> {result.word_count_after}",
          file=sys.stderr)
    print(f"  passes: {len(result.passes)}", file=sys.stderr)
    for note in result.notes:
        print(f"  note: {note}", file=sys.stderr)
    return 0


def cmd_profiles(_args: argparse.Namespace) -> int:
    print("Writing profiles:\n")
    print(describe_profiles())
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    config = load_config()
    if not config.has_key:
        print("MINIMAX_API_KEY is not set.", file=sys.stderr)
        print(config.missing_key_message(), file=sys.stderr)
        return 2
    client = build_client(config, dry_run=args.dry_run)
    ok, message = client.check()
    print(("OK   " if ok else "FAIL ") + message)
    print(f"  model:   {config.model}")
    print(f"  base:    {config.base_url}")
    print(f"  style:   {config.api_style}")
    return 0 if ok else 1


def cmd_models(args: argparse.Namespace) -> int:
    config = load_config()
    try:
        client = build_client(config, dry_run=args.dry_run)
    except MiniMaxError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    models = client.list_models()
    if not models:
        print("Could not list models (the key may not allow it).")
        return 1
    for m in models:
        marker = " <-" if m == config.model else ""
        print(f"  {m}{marker}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Start the Streamlit UI."""
    if not WEB_APP.exists():
        print(f"Error: {WEB_APP} not found.", file=sys.stderr)
        return 2
    config = load_config()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(WEB_APP),
        "--server.port",
        str(args.port or config.streamlit_port),
        "--server.headless",
        "true",
    ]
    if args.open_browser:
        cmd.append("--server.headless=false")
    print(f"Starting the UI on http://localhost:{args.port or config.streamlit_port}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanize",
        description=(
            "Rewrite AI-assisted text so it reads as genuinely human-written. "
            "Run with no arguments to start the web UI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  humanize                        start the web UI\n"
            "  humanize analyze draft.md       score a file, no API calls\n"
            "  humanize rewrite draft.md       rewrite a file\n"
            "  humanize rewrite draft.md -p academic -o out.md\n"
            "  humanize profiles               list the writing profiles\n"
        ),
    )
    parser.add_argument(
        "--port", type=int, default=None, help="port for the web UI (default 8501)"
    )
    parser.add_argument(
        "--open-browser", action="store_true", help="open the UI in a browser"
    )
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="score a text without rewriting it")
    p_analyze.add_argument("file", nargs="?", help="file to analyse, or - for stdin")
    p_analyze.add_argument("-q", "--quiet", action="store_true")
    p_analyze.set_defaults(func=cmd_analyze)

    p_rewrite = sub.add_parser("rewrite", help="rewrite a text")
    p_rewrite.add_argument("file", nargs="?", help="file to rewrite, or - for stdin")
    p_rewrite.add_argument("-o", "--output", help="write the result here")
    p_rewrite.add_argument(
        "-p",
        "--profile",
        default="general",
        choices=profile_keys(),
        help="target writing profile",
    )
    p_rewrite.add_argument("-i", "--instructions", help="extra guidance for the model")
    p_rewrite.add_argument(
        "--passes", type=int, default=None, help="maximum rewrite passes"
    )
    p_rewrite.add_argument(
        "--no-preserve-length", action="store_true", help="do not aim for the same length"
    )
    p_rewrite.add_argument(
        "--no-verify", action="store_true", help="skip the meaning-drift check"
    )
    p_rewrite.add_argument(
        "--dry-run", action="store_true", help="do not call the API"
    )
    p_rewrite.set_defaults(func=cmd_rewrite)

    p_profiles = sub.add_parser("profiles", help="list the writing profiles")
    p_profiles.set_defaults(func=cmd_profiles)

    p_check = sub.add_parser("check", help="verify the API key works")
    p_check.add_argument("--dry-run", action="store_true")
    p_check.set_defaults(func=cmd_check)

    p_models = sub.add_parser("models", help="list available models")
    p_models.add_argument("--dry-run", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_web = sub.add_parser("web", help="start the web UI explicitly")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return cmd_web(args)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except MiniMaxError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
