"""Auto-loaded by pytest before any test module imports — adds the package's
``torch-ext`` source dir to ``sys.path`` so the suite runs without an install
step."""

import sys
from pathlib import Path

_TORCH_EXT = Path(__file__).resolve().parent.parent / "torch-ext"
if str(_TORCH_EXT) not in sys.path:
    sys.path.insert(0, str(_TORCH_EXT))
