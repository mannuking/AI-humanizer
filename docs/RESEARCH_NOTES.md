# Research notes

What was studied, what was measured, and what this implementation does
differently because of it.

**Research date:** 2026-10-31.

---

## 1. The scope correction

There is no product called "Higgs", "Humaniggs" or "human higgs bypass".
An exhaustive search — web, the GitHub API, and the full HumanizerBench
registry of 14 humanizers — found no such product. The registry slugs are:

```
writehuman  stealthgpt  undetectable  clever-ai-humanizer
humanize-ai-pro  ai-humanize-io  super-humanizer  stealth-writer
hix-bypass  walter-writes  supwriter  phrasly  humbot  grammarly
```

The likely intended product is **`blader/humanizer`** — the open-source
"Humanizer" agent skill (MIT), whose architecture is entirely public and
therefore the most implementable reference available. The tool referred to
as "the human higgs bypass" in the original request is treated here as the
Humanizer-class product, and the implementation follows the open-source
skill's pattern catalogue.

---

## 2. What the commercial products actually do

### Walter Writes AI

A purpose-trained commercial model with named modes, its own detector, async
jobs, an MCP server and a preservation report. Its MCP suite includes a
"brand voice adapter" that exposes a rule-based voice specification:

```yaml
voice: [conversational, direct, second-person]
contractions: true
sentence_length: short
register: confident_not_cautious
banned: ["we believe", "perhaps", "it might be worth considering"]
prefer: concrete_nouns
never_use: ["exclamation marks", "game-changer", "best-in-class", "leverage",
            "synergy", "unlock", "supercharge"]
em_dash: max_one_per_paragraph
three_item_lists: prefer_two
```

This schema is directly implementable, so the writing profiles in
`src/humanizer/profiles.py` follow its shape: a register, a rhythm rule, a
paragraph-shape rule, a set of banned or discouraged constructions, and a
per-profile focus list.

### The detection metrics that are actually published

Only one vendor publishes named detection metrics. The rest are opaque
classifiers.

| Metric | Semantics |
| --- | --- |
| `perplexity` | Text predictability. Lower values indicate AI-like patterns. |
| `burstiness` | Sentence-length variation. Human text has higher burstiness. |
| `readability` | Flesch score. AI text has unnaturally consistent scores. |
| `satPercent` | Percentage of advanced/SAT vocabulary words. |
| `simplicity` | Ratio of common words used. |
| `ngramScore` | N-gram pattern detection. Higher values indicate AI fingerprints. |
| `averageSentenceLength` | Mean words per sentence. |

Walter's own detector publishes its bands: 0.0–0.3 human, 0.3–0.5 likely
human, 0.5–0.7 likely AI, 0.7–1.0 AI-generated.

---

## 3. The measured findings that changed the implementation

These are measurements taken across the October 2026 HumanizerBench corpus
(420 test rows, 14 humanizers), not marketing claims.

### 3.1 Every incumbent tool corrupts numbers

Fraction of numeric tokens present in the input that also appear in the
output:

| Tool | Numeric tokens retained |
| --- | --- |
| Undetectable | 0.943 |
| Stealth Writer | 0.818 |
| Walter Writes | 0.807 |
| Phrasly | 0.807 |
| WriteHuman | 0.659 |

Every tool drops 6–35% of the numbers. Walter loses about 19%.

**What this implementation does:** `src/humanizer/preservation.py` extracts
every number, date, percentage, currency amount, citation, URL, code span
and table before the model sees the text, replaces them with opaque
placeholders, restores them afterwards, and then **verifies** that each one
came back. A lost value is reported in the notes panel rather than silently
shipped.

### 3.2 Length inflation is the most common failure

| Tool | Length ratio (output / input) |
| --- | --- |
| Humbot | 1.461 |
| Walter Writes | 1.349 |
| Grammarly | 1.281 |
| Undetectable | 1.228 |

The independent benchmark penalises a ratio above 1.4, and Walter loses nine
penalty points to it.

**What this implementation does:** the rewrite prompt carries a length target
and a per-chunk target, the pipeline reports the final ratio, and a ratio
outside 0.6–1.4× is recorded as a note.

### 3.3 Vocabulary swapping without restructuring is the trap

Mean occurrences per document, Walter's outputs against their inputs:

| Tell | in | out | verdict |
| --- | --- | --- | --- |
| Not X but Y | 0.77 | 0.20 | fixed |
| Forced triads | 1.73 | 0.87 | halved |
| Pivotal / crucial / landscape | 0.23 | 0.03 | eliminated |
| **Furthermore / Moreover / Additionally** | 0.23 | **0.73** | **3× worse** |
| **Shallow "-ing" riders** | 0.90 | **2.00** | **2.2× worse** |
| Curly quotes | 0.27 | 0.80 | worse |
| Em dashes | 1.10 | 1.17 | unchanged |
| Delve / leverage / utilize family | 0.27 | 0.43 | worse |

Walter removes the stock AI words and reduces triad density, but *injects*
transition adverbs and "-ing" riders — both classic AI tells — and inflates
length by 35%.

**What this implementation does:** the rewrite prompt describes the target
register rather than listing banned words, and the detector explicitly
catches the transition adverbs and the "-ing" riders that a swap-based
rewrite tends to introduce. The loop re-scores after every pass, so a
rewrite that fixes one tell while adding another is caught on the next
measurement.

### 3.4 Burstiness is not the lever people think

Mean coefficient of variation of sentence length (SD / mean):

| Tool | input | output | Δ |
| --- | --- | --- | --- |
| StealthGPT | 0.454 | 0.527 | **+0.073** |
| WriteHuman | 0.454 | 0.521 | +0.067 |
| Walter Writes | 0.454 | 0.459 | **+0.005** |

Walter barely moves burstiness despite being marketed as a burstiness tool.
The implementation targets **≥ 0.50 mean CV with no document below 0.35**,
which is where the top performers land.

### 3.5 The best-scoring tool wins by degrading the text

| Rank | Tool | Composite | Bypass | Meaning | Length ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | WriteHuman | 74.89 | 0.888 | 0.600 | 1.091 |
| 12 | Phrasly | 61.79 | **0.953** | **0.499** | 1.019 |
| 13 | Humbot | 61.74 | 0.714 | 0.686 | **1.461** |

Phrasly posts the highest bypass rate with the worst meaning preservation —
it wins by degrading the text. The benchmark's penalty system exists to catch
exactly this, and it is the failure mode to avoid.

**What this implementation does:** a meaning-drift check runs after the first
pass. A drift verdict discards the pass and keeps the input, and the drift is
reported.

---

## 4. The scoring formula used as a reference

The HumanizerBench composite, the only published scoring formula in the
ecosystem:

```
composite_raw = 42 * bypass_rate
              + 32 * meaning_preservation
              + 16 * readability
              + 10 * consistency_across_categories
composite     = max(0, composite_raw - sum(penalties))
```

Its real-rewrite gate, which this implementation follows in spirit:

```
real_rewrite = 0.6 <= wc(out)/wc(in) <= 1.4
               AND meaning_preservation >= 0.85
```

**Bypass is 42% of the score, meaning preservation is 32%.** A tool that
optimises detection score alone is optimising less than half the objective —
which is why this implementation treats meaning preservation as a hard gate
rather than a tiebreaker.

---

## 5. The pattern catalogue

The catalogue in `src/humanizer/data.py` is generated by
`scripts/convert_reference_patterns.py` from two open-source humanizer
engines and the Wikipedia guide they are built on.

**A note on the source's own numbers.** The upstream `SKILL.md` says "28
patterns" and its `README.md` says "24", but the code ships **30** — the code
is authoritative. Its "500+ vocabulary terms" claim is also not reproducible:
there are 282 unique tier words and 109 phrase regexes, which is 391 lexicon
items, not 500. This implementation uses the measured numbers.

It contains:

- **30 named detectors** in five categories — content, language, style,
  communication, filler — each with a weight and a description
- **282 tiered vocabulary terms** — tier 1 is the unmistakable vocabulary
  (delve, tapestry, vibrant, realm, seamless), tier 3 is the generic
  (significant, crucial, robust)
- **109 stock phrases** with the regex that detects them and a fix hint

The generator handles the JavaScript/Python dialect differences:

| Issue | Handling |
| --- | --- |
| JS astral code points (`\u{1F300}`) | Converted to Python's `\U0001F300` |
| Character classes containing `/` | The literal body is scanned past the class |
| JS division mistaken for a regex | Rejected by context heuristics and by a compile check |
| Block-comment stripping eating regex literals | Only line-initial block comments are stripped |

All 241 regexes are verified to compile under Python's `re` module.

---

## 6. The scoring formula in this implementation

The tool's own score is deliberately simpler than the benchmark composite,
because it does not have access to six commercial detectors:

```
pattern_score = saturate(weighted_findings_per_100_words / 40)
style_score   = 0.30*burstiness_uniformity
              + 0.20*type_token_repetitiveness
              + 0.15*hapax_deficit
              + 0.15*punctuation_uniformity
              + 0.10*em_dash_density
              + 0.10*paragraph_uniformity

overall_score = 0.65 * pattern_score + 0.35 * style_score
```

The lexical layer dominates deliberately. A passage that says "delve into the
intricate tapestry of learning" is machine-written whatever its sentence
rhythm, and a purely statistical score would miss it.

**This score is a style estimate, not a probability of authorship.** The
benchmark treats every vendor score the same way, and the tool says so in its
own output.

---

## 7. Open items

- The `blader/humanizer` pattern catalogue (26 patterns in the skill, 24 in
  the earlier release) overlaps substantially with the 30-detector catalogue
  ported here. Merging them would add the document-scale patterns — a
  contrast split across two sentences, an identical opening on every section
  — which the current engine only partly covers.
- The HumanizerBench readability sub-score is produced by an LLM rating. This
  implementation uses the Flesch score instead, which is cheaper but a
  different measurement.
- The scoring thresholds here are calibrated against the reference
  implementations, not against a labelled corpus of human writing. A
  calibration set would tighten them.
