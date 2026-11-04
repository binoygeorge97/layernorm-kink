"""SHA-256 of the parameter arrays of a controller npz (not of the file: npz/zip carry timestamps).
Digest over the sorted keys: key, dtype, shape and the raw C-order bytes of each array.
Usage: python tools/hash_params.py FILE [FILE ...]"""
import hashlib
import sys
import numpy as np


def param_hash(fn):
    z = np.load(fn, allow_pickle=False)
    h = hashlib.sha256()
    for k in sorted(z.files):
        a = np.ascontiguousarray(z[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(param_hash(f), f)
