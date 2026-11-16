# Movie Recommender with Matrix Factorization

Predict missing movie ratings with low-rank matrix factorization, trained by
alternating least squares (ALS), built from scratch with NumPy.

## Repository layout

```
src/       load_data.py (load + describe MovieLens), simulation.py (known-truth check)
data/      raw data goes here (not committed)
figures/   plots written by the scripts
docs/      Overleaf report (iteration2_report.tex / .pdf)
```

## Setup

Only `numpy`, `pandas`, `matplotlib` and `scipy` are used.

```
pip install numpy pandas matplotlib scipy
```

## Getting the data

MovieLens 100K is not committed (add `data/` to `.gitignore`).

1. Download `ml-100k.zip` from https://grouplens.org/datasets/movielens/100k/
2. Unzip it so the file sits at `data/ml-100k/u.data`

## Running

From the repo root:

```
python src/simulation.py   # Iteration 2 simulation check, writes figures/sim_*.png
python src/load_data.py    # dataset description + figures/rating_hist.png etc.
```

Seeds are fixed with `np.random.default_rng(0)`, so everyone gets the same numbers.

## Overleaf

Upload `docs/iteration2_report.tex` and the `figures/` folder, keeping the
folder structure, or copy the PNGs next to the .tex file.

## Task list

| ID | Task | Owner | Checker | Status |
|----|------|-------|---------|--------|
| T1 | Notation and definitions | Hakim | | done |
| T2 | Assumptions | Hakim | | done |
| T3 | Q1-Q3 derivations | Hakim | | done |
| T4 | Q4-Q5 derivations + MAP bonus | Hakim | | done |
| T5 | Algorithm as math | Hakim | | done |
| T6 | Real data loading and description | Hakim | | run on real data |
| T7 | Simulation check | Hakim | | done |
| T8 | Check all derivations | | | open |
