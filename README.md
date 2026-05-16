# Bilinear SPD Benchmark

Do rank-one parameter decompositions recover bilinear functional mechanisms?

This repo trains a tiny bilinear MLP on modular addition, extracts the exact
probe-conditioned interaction matrix `M_r` and its eigenspaces, learns an
SPD-style rank-one gated parameter decomposition, and asks whether the learned
atoms (or clusters of co-active atoms) align with the bilinear eigenspaces.

The theory writeup and codebase spec that drive this repo are kept locally
(not committed). `summary.md` rolls up what's been built and verified so far.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Day 1: train the bilinear MLP
python -m scripts.train_bilinear configs/modadd_p13.yaml

# Sanity tests
pytest
```

Runs land in `runs/<run_name>/`.

## Layout

```
src/bilinear_spd/   library code
scripts/            CLI entry points
configs/            YAML experiment configs
tests/              pytest suite
runs/               output artifacts (gitignored)
docs/               project spec + theory + style guide
```
