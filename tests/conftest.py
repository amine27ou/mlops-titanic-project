import sys
from pathlib import Path

# Make src available to pytest so tests can import project packages.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
