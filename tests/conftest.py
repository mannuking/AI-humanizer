"""Test configuration.

Makes the package importable when the tests are run from the project root
without an editable install, and keeps the pytest cache inside the project
rather than the user's home directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
