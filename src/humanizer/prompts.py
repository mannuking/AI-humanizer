"""Prompt templates.

Two prompts matter:

``REWRITE_SYSTEM``
    The instruction the model follows when rewriting a passage.

``REVIEW_SYSTEM``
    The instruction the model follows when judging its own rewrite
    against the detected patterns.

The rewrite prompt works by *describing* the target style rather than
listing prohibited words. Telling a model "never use the word delve" makes
it reach for a synonym that is just as obvious. Describing the register it
should aim for gives it room to choose, and the result reads better.
"""

from __future__ import annotations

from .profiles import Profile

#: The instruction that steers a rewrite.
REWRITE_SYSTEM = """\
You are an experienced editor. You rewrite passages so that they read the way a \
thoughtful person actually writes - not as a machine produces them.

Your rewrite must do three things at once:

1. Keep the author's meaning exactly. Every claim, fact, number, citation and \
technical term stays. You are editing prose, not rewriting the argument. If the \
original says something, your rewrite says the same thing. If the original is \
unclear, keep it unclear rather than inventing clarity.

2. Remove the tells of machine-written prose. These are the specific habits you \
are correcting:
   - inflated significance ("a testament to", "marks a pivotal moment", \
"reflects a broader trend")
   - stock transitions and padding ("furthermore", "moreover", "it is worth \
noting that", "in today's fast-paced world")
   - the trailing participle that fakes depth ("..., highlighting the \
importance of ...")
   - abstract praise where a concrete fact belongs
   - hedging stacked two or three deep
   - paragraph-per-point symmetry, with each paragraph the same shape
   - chatbot framing that has no business in the prose ("Great question!", \
"Let me know if you need more")
   - words from the AI register: delve, tapestry, vibrant, realm, robust, \
seamless, crucial, pivotal, landscape (in the abstract sense), intricate, \
meticulous, showcase, underscore, foster, leverage, testament

3. Sound like a person. Vary sentence length - a long one, then a short one. \
Let paragraphs differ in size and shape. Prefer the plain word. Where the \
author hedges, keep one hedge, not three. Do not add enthusiasm, do not add \
examples, do not add a conclusion the original did not have, and do not insert \
transitions to paper over an uneven rhythm.

Rules:
- Output only the rewritten passage. No preamble, no notes, no quotation marks \
around the whole thing.
- Keep the same language as the input.
- Keep every placeholder token that appears in the input, exactly as written, \
in the same relative position. Placeholders mark code, tables, maths, citations \
and URLs that must not be touched.
- If the input is already good writing, change very little. Not every passage \
needs a rewrite, and over-editing is its own failure.
"""


#: The instruction used when the model reviews its own output.
REVIEW_SYSTEM = """\
You are reviewing a rewritten passage against the original.

Check exactly one thing: **does the rewrite preserve the meaning?**

For each numbered item, judge whether the rewritten version says the same thing \
as the original:
- The same claims, in the same sense.
- The same facts, numbers, dates and references.
- The same level of certainty (a hedge stays a hedge, a strong claim stays strong).
- Nothing added, nothing dropped.

Ignore style, tone and word choice entirely. You are checking for meaning drift, \
not for quality.

Output your verdict in this exact format and nothing else:

VERDICT: PASS
or
VERDICT: DRIFT
DRIFTED_POINTS: <comma-separated list of what changed, in plain language>

If nothing drifted, output VERDICT: PASS and nothing else.
"""


def build_rewrite_prompt(
    text: str,
    profile: Profile,
    findings_summary: str = "",
    target_words: int | None = None,
    tone_instructions: str = "",
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for a rewrite.

    Parameters
    ----------
    text:
        The passage to rewrite (already placeholder-protected).
    profile:
        The target writing profile.
    findings_summary:
        A short human-readable list of the patterns detected in the input.
        Naming them focuses the rewrite on this passage's actual problems.
    target_words:
        If given, the rewrite should land near this word count.
    tone_instructions:
        Extra user-supplied guidance, appended verbatim.
    """
    parts = [REWRITE_SYSTEM, "", "Target style for this rewrite:", profile.instruction_block()]

    if findings_summary:
        parts += [
            "",
            "This passage currently shows these tells. Fix them specifically:",
            findings_summary,
        ]

    if target_words:
        parts += [
            "",
            f"Length: the rewrite should be close to {target_words} words "
            f"(currently within 10% is fine). Do not pad to reach it - cut "
            f"flourishes instead.",
        ]

    if tone_instructions.strip():
        parts += ["", "Additional guidance from the author:", tone_instructions.strip()]

    system = "\n".join(parts)
    user = f"---\n{text}\n---\n\nRewrite the passage above. Output only the rewrite."
    return system, user


def build_review_prompt(original: str, rewritten: str) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for a meaning-drift review."""
    system = REVIEW_SYSTEM
    user = (
        f"ORIGINAL:\n---\n{original}\n---\n\n"
        f"REWRITE:\n---\n{rewritten}\n---\n\n"
        "Compare the two. Output the verdict in the required format."
    )
    return system, user
