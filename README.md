
## What it actually does

The tools in this category mostly swap one AI word for another. That buys a
low perplexity score while leaving the text structurally machine-shaped, and
the newer detectors catch exactly that.

This one works in layers:

```
  analyse  →  protect  →  rewrite  →  restore  →  verify  →  re-score
                            ↑                               │
                            └─────── not good enough ───────┘
```

1. **Analyse** — 30 named detectors look for the tells (inflated significance,
   stock transitions, trailing "-ing" riders, chatbot residue, formatting
   crutches), plus a statistical pass over sentence rhythm and vocabulary
   variety. You get a score and a list of what to fix.
2. **Protect** — numbers, dates, percentages, citations, code, URLs, tables and
   emphasis are hidden behind opaque placeholders so the model cannot damage
   them.
3. **Rewrite** — the passage is rewritten in blocks, with the detected tells
   named in the instructions and a target style profile applied.
4. **Restore** — the protected values go back, and then they are *verified*.
5. **Verify** — a second call checks that the rewrite says the same thing as
   the original. If it drifted, the pass is discarded and you keep the input.
6. **Re-score** — if the score is still high, another pass runs, now knowing
   what the last one missed.

### The two things it gets right that the commercial tools do not

**Numbers survive.** Measured across the commercial humanizers, every one of
them corrupts or drops between 6% and 35% of the numeric tokens in the input;
one of them loses about 19%. That failure is invisible in the marketing and it
is the easiest thing to fix — hide the spans, put them back, then check they
came back. This tool reports any value that did not survive.

**Length stays honest.** The independent benchmark gates a rewrite at 0.6–1.4×
the input length. The best-scoring commercial tool averages 1.35× and loses
nine penalty points for it. This tool targets 1.0× and reports the ratio.

---

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mannuking/AI-humanizer.git
cd AI-humanizer

uv sync                       # create .venv and install everything
cp .env.example .env          # then add your key
```

Add your MiniMax key to `.env`:

```bash
MINIMAX_API_KEY=your-key-here
MINIMAX_MODEL=MiniMax-M3
```

The key can also come from the environment. `MINIMAX_API_KEY` is read from
`.env` in the project root or from the process environment.

### Two commands, as requested

| Command | What it does |
| --- | --- |
| `uv sync` | Creates `.venv` and installs the package and its dependencies |
| `uv run humanize` | Starts the app (Streamlit UI on port 8501) |

`uv run humanize` with no arguments launches the web UI. Everything else is a
subcommand.

---

## Command line

```bash
uv run humanize                          # start the web UI
uv run humanize analyze draft.md         # score a file, no API calls
uv run humanize rewrite draft.md         # rewrite a file
uv run humanize rewrite draft.md -p academic -o out.md
uv run humanize rewrite draft.md -i "keep the first person voice"
uv run humanize profiles                 # list the writing profiles
uv run humanize check                    # verify the API key and model
uv run humanize models                   # list the models the key can use
```

`rewrite` reads stdin when the file argument is `-`.

---

## Writing profiles

The profile is the target style. It sets the register, the sentence rhythm, the
paragraph shape, and the tells that profile is especially keen to remove.

| Key | Profile | For |
| --- | --- | --- |
| `general` | General / professional | Reports, emails, blog posts — the default |
| `academic` | Academic / research paper | Journal prose: hedged, cited, formal |
| `conversational` | Conversational | A colleague explaining something |
| `technical` | Technical / documentation | Commands, specs, engineering notes |
| `narrative` | Narrative / essay | Reviews, essays, anything with a point of view |

`technical` will not touch a command, a flag, a path or a version number.
`academic` will not add a summary paragraph the original did not have.

---

## The web UI

- Side-by-side source and result
- Live score breakdown: pattern density, style uniformity, and the full
  statistical profile
- A table of every detected pattern with the line it is on and what to do
  about it
- Download button and copy-to-clipboard
- A notes panel that reports protected values lost, length drift, and any
  meaning drift that was caught and reverted

---

## Configuration

Everything is read from `.env` or the environment.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MINIMAX_API_KEY` | — | **Required.** The API key. |
| `MINIMAX_MODEL` | `MiniMax-M3` | Model id. Pin `MiniMax-M3` or `MiniMax-M3.1-Flash`; the M2.x family retires on 2026-11-18. |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | API base. |
| `MINIMAX_API_STYLE` | `openai` | `openai` or `anthropic`. MiniMax serves both. |
| `MINIMAX_TIMEOUT` | `120` | Request timeout, seconds. |
| `MINIMAX_MAX_RETRIES` | `3` | Retries on a transient failure. |
| `HUMANIZER_TEMPERATURE` | `0.85` | Sampling temperature. |
| `HUMANIZER_TOP_P` | `0.95` | Nucleus sampling. |
| `HUMANIZER_MAX_TOKENS` | `8192` | Response cap. |
| `HUMANIZER_MAX_PASSES` | `3` | Rewrite passes before giving up. |
| `HUMANIZER_CHUNK_CHARS` | `6000` | Characters per rewrite block. |
| `HUMANIZER_TARGET_SCORE` | `0.25` | Stop rewriting below this score. |
| `HUMANIZER_PRESERVE_WORD_COUNT` | `true` | Aim for the same length. |
| `HUMANIZER_WORD_COUNT_TOLERANCE` | `0.08` | Allowed length drift. |
| `STREAMLIT_PORT` | `8501` | UI port. |

---

## How it works

### The detection catalogue

**40 detectors** across three layers, plus 282 tiered vocabulary terms and
109 stock-phrase patterns:

| Layer | What it catches |
| --- | --- |
| **Lexical** (30) | Phrase-level and formatting tells: inflated significance, stock transitions, shallow "-ing" riders, chatbot residue, bold-as-decoration, curly quotes |
| **Threshold** (10) | Patterns that are only a tell when they repeat: em-dash density, three-or-more bold spans, title-case headings, inline-header lists, over-structured content, synonym cycling |
| **Document** (10) | Patterns a single-sentence regex cannot see: a one-line closing paragraph, a contrast split across two sentences, repeated sentence openings, a staged run-up, a heading repeated in the first line |

The catalogue lives in
[`src/humanizer/data.py`](src/humanizer/data.py) and is generated by
[`scripts/convert_reference_patterns.py`](scripts/convert_reference_patterns.py)
from the reference implementations, so the trigger strings are the real ones
rather than paraphrases. The threshold and document layers live in
[`thresholds.py`](src/humanizer/thresholds.py) and
[`document.py`](src/humanizer/document.py).

### The context guardrails

A watched phrase inside a quotation, a heading, a code span, a link target or
a citation is not the writer using it — it is the writer discussing it. The
catalogue is explicit about this, and it is the difference between a detector
that reads like a person and one that gets ignored:

> Leave a watched phrase alone inside a quotation, a title, a proper name, or
> a passage that discusses the phrase rather than uses it.

So those regions are found before anything is scored, and matches inside them
are suppressed. The same regions are protected from the rewrite, so a quoted
passage comes back exactly as it went in.

The one exception is the curly-quote detector, which reports on the quote
characters themselves and is therefore never suppressed.

### The statistical layer

| Metric | What it measures | Why it matters |
| --- | --- | --- |
| Burstiness | Coefficient of variation of sentence length | Human prose varies; LLM output converges |
| Type-token ratio | Distinct words / total words | Repetition reads as machine-made |
| Hapax ratio | Share of words that appear exactly once | Vocabulary that repeats is a tell |
| Punctuation entropy | Spread of punctuation usage | Uniform punctuation is uniform prose |
| Reading ease | Flesch score | Useful as a *consistency* signal, not a quality one |

The combined score is 65% lexical and 35% statistical. The lexical layer
dominates on purpose: a sentence that says "delve into the intricate tapestry"
reads as machine-written whatever its rhythm. Document-scale findings count at
half weight — they are weaker signals alone, but several together matter.

### The scoring band

| Score | Reading |
| --- | --- |
| 0.00 – 0.15 | Low — reads naturally |
| 0.15 – 0.35 | Moderate — some AI tells |
| 0.35 – 0.60 | High — strong AI signature |
| 0.60 – 1.00 | Very high — pervasive AI signature |

---

## What it does not do

- **It does not insert typos, shuffle sentences or mangle spelling.** Those
  tricks damage the text and are still caught. They also make the result
  worse as writing.
- **It does not invent facts.** If a sentence needs a detail the original did
  not have, the instruction is to write a simpler sentence instead.
- **It does not rewrite quotations, titles or proper names.**
- **It does not prove authorship.** A score is an estimate of style, not
  evidence about who wrote a text. The benchmark gates treat any score as
  "not a calibrated probability of authorship", and so does this.

---

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q      # 92 tests, no API key needed
uv run ruff check src/ tests/ scripts/
```

The test suite runs entirely offline: the LLM layer is exercised through a
stub backend, and everything else is pure.

---

## Provenance

The pattern catalogue and the writing-profile design were derived from
public reference implementations and published research:

- **Wikipedia — "Signs of AI writing"** and **WikiProject AI Cleanup**, the
  guide Wikipedia editors use to catch machine-written text.
- **`blader/humanizer`** and **`brandonwise/humanizer`**, open-source
  humanizer engines whose pattern catalogues and statistical methods are
  public.
- The **HumanizerBench** independent benchmark, for the scoring bands, the
  length gates and the meaning-preservation thresholds.
- **Walter Writes AI** and **StealthWriter**, for the commercial architecture
  (profile modes, multi-pass loops, preservation reports).

See [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) for the full write-up
with the measured numbers.

---

## Credits

The detection catalogue, the writing-profile design and the scoring approach
come from public research and open-source implementations — see
[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) for the sources and the
measurements behind each decision.

Created with ❤️ by JAI.

---

## License

MIT — see [`LICENSE`](LICENSE).
