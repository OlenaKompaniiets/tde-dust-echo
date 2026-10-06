"""UGC11487 V4_FIXED numerical implementation."""
import numpy as np
# NumPy 1.26 (common on Python 3.11) calls the same routine trapz.
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
