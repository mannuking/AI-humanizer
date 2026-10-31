"""AI-humanizer: rewrite AI-assisted text so it reads as genuinely human-written.

The package is deliberately layered so each concern is testable in isolation:

    humanizer.patterns   the AI-writing signature catalogue (what to look for)
    humanizer.structure  document parsing that survives a round trip
    humanizer.analyze    statistical + pattern scoring of a passage
    humanizer.profiles   writing "modes" and voice profiles
    humanizer.llm        the MiniMax client (and a stub backend for tests)
    humanizer.prompts    the instruction templates sent to the model
    humanizer.pipeline   the multi-pass rewrite loop
    humanizer.cli        `humanize` entry point
    humanizer.web        the Streamlit interface
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
