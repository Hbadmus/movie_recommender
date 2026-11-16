# Movie Recommender with Matrix Factorization

**Team:** Hakim Badmus, Doyin Okwesa

We predict missing movie ratings by splitting the rating matrix into a user matrix
and a movie matrix (matrix factorization). We train it with alternating least
squares (ALS), written from scratch in NumPy.

## Folders

```
src/       Python code
data/      the dataset goes here (not uploaded to GitHub)
figures/   plots made by the code
docs/      Overleaf report (PDF + .tex) and Excel project plan
```

## Setup

We only use numpy, pandas, matplotlib and scipy.

```
pip install numpy pandas matplotlib scipy
```

## Getting the data

The dataset is too big to keep in the repo, so download it yourself:

1. Download `ml-100k.zip` from https://grouplens.org/datasets/movielens/100k/
2. Unzip it into the `data/` folder so this file exists: `data/ml-100k/u.data`

## Running the code

Run these from the main repo folder:

```
python src/load_data.py     # describes the dataset, saves rating_hist.png and sparsity.png
python src/simulation.py    # simulation check, saves sim_objective.png
```

The random seed is fixed (`np.random.default_rng(0)`), so everyone gets the same numbers.

## What each script does

- **load_data.py**: loads MovieLens 100K, prints the size, column types, missing values
  and rating counts, splits the ratings 80/20 into train and test, and makes two plots.
- **simulation.py**: builds a fake rating matrix where we know the true answer (rank 3),
  hides 70% of it, runs ALS, and checks the results from our derivations (Q1 to Q5).

## Report

The full math is in `docs/iteration2_report.pdf`: notation, definitions, assumptions,
derivations for Q1 to Q5, a bonus probability result, the algorithm, the data
description and the simulation check.

## Task list

| ID | Task | Owner | Checker | Status |
|----|------|-------|---------|--------|
| T1 | Notation and definitions | Hakim | Doyin | done |
| T2 | Assumptions | Doyin | Hakim | done |
| T3 | Q1: objective | Hakim | Doyin | done |
| T4 | Q2: solve for one user | Hakim | Doyin | done |
| T5 | Q3: positive definite | Hakim | Doyin | done |
| T6 | Q4: ALS never increases J | Doyin | Hakim | done |
| T7 | Q5: SVD vs. missing data | Doyin | Hakim | done |
| T8 | Bonus: where lambda comes from | Doyin | Hakim | done |
| T9 | Algorithm as math | Hakim | Doyin | done |
| T10 | Data loading and description | Hakim | Doyin | done |
| T11 | Simulation check | Hakim | Doyin | done |
| T12 | Excel project plan | Hakim | Doyin | to do |
