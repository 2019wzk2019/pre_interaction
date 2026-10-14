"""Entry: TP deep-learning pipeline. Does not train unless --train is passed."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grip_dl.train import main

if __name__ == "__main__":
    main()
