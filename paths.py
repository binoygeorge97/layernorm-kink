"""Repository paths (Task 0, CLAUDE.md section 3.3). Replaces the /home/claude/... paths of the analysis sandbox.

REPO  repository root
SRC   experiment modules (level1_core, dpc_core, phase_core, ...); formerly /home/claude/kinkwork
DATA  the dpc_out data folder: env LK_DATA, else <REPO>/data/dpc_out (formerly /home/claude/data/dpc_out)
OUT   outputs of re-runs: env LK_OUT, else <REPO>/out (never inside an earlier result folder)

Scripts in sub-folders find this file with
    import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents
                                                     if (p / "paths.py").exists()))); import paths
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "src")
DATA = os.path.abspath(os.environ.get("LK_DATA", os.path.join(REPO, "data", "dpc_out")))
OUT = os.path.abspath(os.environ.get("LK_OUT", os.path.join(REPO, "out")))
REF = os.path.join(REPO, "data", "reference_results")


def out(*parts):
    """<OUT>/parts..., creating the directory (the last part is treated as a file name if it has a suffix)."""
    p = os.path.join(OUT, *parts)
    d = p if not os.path.splitext(p)[1] else os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    return p


def find(name, *dirs):
    """first existing <dir>/<name> among dirs (used to read inputs that a re-run writes under OUT but that also ship
    with the repository, e.g. story/swap_lqr_rates.csv)."""
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {dirs}")


if SRC not in sys.path:
    sys.path.insert(0, SRC)
