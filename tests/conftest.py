import sys
from pathlib import Path


# Allow tests to import project modules like `env.*`, `config.*`, `utils.*`
# when running via `pytest` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

