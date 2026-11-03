# layernorm-kink: experiments for the paper

How the pre-norm LayerNorm Jacobian "kink" in a neural surrogate reaches model-based controllers (fixed and rescheduled LQR,
differentiable predictive control). Owner: Binoy George.

Start here: `CLAUDE.md` (full handout: what is established, repository map, conventions, pitfalls, Tasks 0-5).

Quick start
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git init && git add -A && git commit -m "Handoff snapshot 2026-11-03"
# replace data/dpc_out with the full MyDrive/dpc_out folder (see data/MANIFEST.md)
cd theory && T2_DATA=../data/dpc_out T2_OUT=../out/theory python verify_thm1.py
```

Layout: `src/` experiment code and Colab notebooks, `theory/` Theorem 1/2 scripts, `story/` slide-figure scripts,
`companion/` LaTeX companions and figure scripts, `docs/` notes, `data/` partial data + reference results,
`analysis/`, `prereg/`, `out/` for new work.
