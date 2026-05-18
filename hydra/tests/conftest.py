from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_EXT = ROOT / "torch-ext"
sys.path.insert(0, str(TORCH_EXT))
