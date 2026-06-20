import os
import sys
from pathlib import Path

# Add src to python path so slurmify is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Enable mock mode universally for tests
os.environ["SLURMIFY_MOCK"] = "1"
