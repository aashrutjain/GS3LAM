"""Ensures the repo root is on sys.path so `from src.cbf... import ...` and
`from eval_cbf_modes import ...` resolve under plain `pytest` (not just
`python -m pytest`) -- there's no pyproject.toml/setup.cfg installing this
project as a package, so pytest's default "prepend" import mode would
otherwise only add tests/ itself, not repo root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
