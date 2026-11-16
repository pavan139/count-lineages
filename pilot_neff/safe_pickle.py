"""Load a pickle that contains only plain containers and numeric/string numpy arrays.
Anything else in the file raises instead of being imported."""
import pickle, sys
import numpy as np
from numpy._core import multiarray as _ma

ALLOWED = {("numpy.core.multiarray", "_reconstruct"): _ma._reconstruct, ("numpy._core.multiarray", "_reconstruct"): _ma._reconstruct,
           ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"blocked import {module}.{name}")


def describe(o, depth=0, key="root"):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key}: dict[{len(o)}] keys={list(o)[:12]}")
        for k in list(o)[:12]:
            describe(o[k], depth + 1, str(k))
    elif isinstance(o, np.ndarray):
        extra = ""
        if o.dtype.kind == "f":
            extra = f" nan={float(np.isnan(o).mean()):.3f} unique~{np.unique(o[~np.isnan(o)][:200000])[:6]}"
        print(f"{pad}{key}: ndarray {o.dtype} {o.shape}{extra}")
        if o.dtype.kind == "U":
            print(f"{pad}  e.g. {list(o[:4])}")
    else:
        print(f"{pad}{key}: {type(o).__name__} {str(o)[:80]}")


def load(path):
    with open(path, "rb") as f:
        return Restricted(f).load()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("==", p); describe(load(p))
