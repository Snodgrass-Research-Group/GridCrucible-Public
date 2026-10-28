"""The three nested QNCA loops (composition members of GridCrucible).

OHM (inner, ohm.py) · TUNE (middle, tune.py) · TRIM (outer, trim.py).
See docs/ARCHITECTURE.md §1 for the loop → MATLAB mapping.
"""

from .ohm import OHMLoop, OHMResult
from .trim import TRIMLoop, TrimMarking, TrimResult
from .tune import TUNELoop

__all__ = ["OHMLoop", "OHMResult", "TRIMLoop", "TrimMarking", "TrimResult", "TUNELoop"]
