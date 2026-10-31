"""Writing profiles - the modes a rewrite can target.

The commercial tools in this space differ mainly in *which* transformation
profile they apply, not in whether they apply one. Each profile below is a
set of concrete, checkable instructions: sentence rhythm, vocabulary
register, paragraph shape, and the tells that must go.

A profile is a description of a target style, not a set of obfuscation
tricks. Nothing here inserts fake typos, shuffles sentences out of order,
or corrupts meaning - those strategies destroy the text and are the reason
most detectors still catch heavily "humanised" output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """A named transformation profile."""

    key: str
    label: str
    description: str
    register: str
    rhythm: str
    paragraph_shape: str
    #: Instructions appended to the rewrite prompt.
    instructions: tuple[str, ...] = ()
    #: Patterns this profile is especially keen to remove.
    focus: tuple[str, ...] = ()

    def instruction_block(self) -> str:
        """Render the profile as prompt instructions."""
        lines = [
            f"Target register: {self.register}",
            f"Sentence rhythm: {self.rhythm}",
            f"Paragraph shape: {self.paragraph_shape}",
        ]
        if self.focus:
            lines.append("Pay particular attention to removing: "
                         + ", ".join(self.focus) + ".")
        lines.extend(self.instructions)
        return "\n".join(f"- {line}" for line in lines)


ACADEMIC = Profile(
    key="academic",
    label="Academic / research paper",
    description=(
        "Formal, hedged, citation-aware prose. The register of a journal "
        "article rather than a blog post."
    ),
    register=(
        "formal scholarly English with the discipline's own terminology; "
        "measured claims, explicit attribution, no marketing language"
    ),
    rhythm=(
        "long analytic sentences broken by short declaratives; the length "
        "varies from clause to clause rather than following a template"
    ),
    paragraph_shape=(
        "one claim per paragraph, developed with evidence; topic sentences "
        "that state the finding rather than announcing it"
    ),
    instructions=(
        "Keep every citation, number, technical term and claim intact.",
        "Use the hedging the discipline actually uses, and no more.",
        "Prefer the specific noun over the abstract one.",
        "Do not add a summary paragraph that the original did not have.",
    ),
    focus=(
        "significance inflation",
        "vague attributions",
        "formulaic challenges",
        "filler phrases",
        "generic conclusions",
    ),
)

GENERAL = Profile(
    key="general",
    label="General / professional",
    description=(
        "The default. Clean, readable prose that suits a report, an email "
        "or a blog post."
    ),
    register=(
        "plain modern English; the register of a competent professional "
        "writing to other professionals"
    ),
    rhythm=(
        "sentence length varies with the content; short sentences carry the "
        "point, longer ones carry the reasoning"
    ),
    paragraph_shape=(
        "paragraphs sized to their idea - two sentences if the idea is two "
        "sentences long"
    ),
    instructions=(
        "Prefer the shortest word that carries the meaning.",
        "Cut hedges that do not change the meaning of the claim.",
        "Keep the author's argument and every fact.",
    ),
    focus=(
        "AI vocabulary",
        "significance inflation",
        "filler phrases",
        "superficial -ing analyses",
    ),
)

CONVERSATIONAL = Profile(
    key="conversational",
    label="Conversational",
    description=(
        "Warm and direct, the way a person explains something to a "
        "colleague or a reader of a blog."
    ),
    register=(
        "conversational English with contractions and first-person "
        "references where the author would actually use them"
    ),
    rhythm=(
        "short-to-medium sentences with occasional long ones; contractions "
        "and fragments where a person would use them"
    ),
    paragraph_shape=(
        "loose paragraphs that follow the thought rather than a fixed grid"
    ),
    instructions=(
        "Use contractions where natural.",
        "Do not manufacture enthusiasm or filler enthusiasm phrases.",
        "Keep every fact and the original point of view.",
    ),
    focus=(
        "sycophantic tone",
        "filler phrases",
        "generic conclusions",
        "chatbot artifacts",
    ),
)

TECHNICAL = Profile(
    key="technical",
    label="Technical / documentation",
    description=(
        "Precise documentation prose: instructions, specifications, "
        "engineering notes."
    ),
    register=(
        "exact technical English; the vocabulary of the domain, used "
        "precisely rather than decoratively"
    ),
    rhythm=(
        "direct sentences for instructions; longer sentences for the "
        "reasoning behind them"
    ),
    paragraph_shape=(
        "one topic per block; steps numbered, caveats stated where they "
        "apply"
    ),
    instructions=(
        "Never alter a command, a flag, a path, a version number or a "
        "code identifier.",
        "Keep the order of steps exactly as written.",
        "Do not add steps that were not in the original.",
    ),
    focus=(
        "copula avoidance",
        "excessive structure",
        "inline-header lists",
        "rule of three",
    ),
)

NARRATIVE = Profile(
    key="narrative",
    label="Narrative / essay",
    description=(
        "Reflective prose with a personal voice - a review, an essay, a "
        "report with a point of view."
    ),
    register=(
        "reflective prose with an audible author; specific and grounded "
        "rather than abstract"
    ),
    rhythm=(
        "rhythm follows the argument; sentence lengths move with the "
        "emotional weight of the point"
    ),
    paragraph_shape=(
        "paragraphs that turn - start on one thought and end on another"
    ),
    instructions=(
        "Keep the author's own judgements; do not neutralise them.",
        "Replace abstractions with the concrete thing being described.",
    ),
    focus=(
        "generic conclusions",
        "filler phrases",
        "significance inflation",
        "excessive hedging",
    ),
)

PROFILES: dict[str, Profile] = {
    p.key: p for p in (ACADEMIC, GENERAL, CONVERSATIONAL, TECHNICAL, NARRATIVE)
}

DEFAULT_PROFILE = "general"


def get_profile(key: str) -> Profile:
    """Look up a profile by key, falling back to the general one."""
    return PROFILES.get((key or "").strip().lower(), PROFILES[DEFAULT_PROFILE])


def profile_keys() -> list[str]:
    return list(PROFILES)


def describe_profiles() -> str:
    """A short human-readable listing, for the CLI and the UI."""
    return "\n".join(
        f"  {p.key:<15} {p.label}\n"
        f"  {'':<15} {p.description}"
        for p in PROFILES.values()
    )
