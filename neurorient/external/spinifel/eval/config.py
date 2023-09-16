try:
    import cupy as xp
    from cupyx.scipy import ndimage
except ImportError:
    import numpy as xp
    from scipy import ndimage
