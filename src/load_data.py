# Load MovieLens 100K and describe it.
# Needs data/ml-100k/u.data (see README).
# Run from the repo root:  python src/load_data.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# each row of u.data: user id, movie id, rating, timestamp (tab separated)
df = pd.read_csv("data/ml-100k/u.data", sep="\t",
                 names=["user", "movie", "rating", "timestamp"])

# make ids start at 0 so they can be used as matrix indexes
df["user"] = df["user"] - 1
df["movie"] = df["movie"] - 1

n_users = df["user"].max() + 1
n_movies = df["movie"].max() + 1

print("number of ratings:", len(df))
print("columns and types:")
print(df.dtypes)
print("missing values:", df.isna().sum().sum())
print("users x movies:", n_users, "x", n_movies)
print("fraction of matrix observed:", round(len(df) / (n_users * n_movies), 4))
print("average rating:", round(df["rating"].mean(), 3))
print("count of each rating:")
print(df["rating"].value_counts().sort_index())

per_user = df.groupby("user").size()
per_movie = df.groupby("movie").size()
print("ratings per user  (min, median, max):", per_user.min(), per_user.median(), per_user.max())
print("ratings per movie (min, median, max):", per_movie.min(), per_movie.median(), per_movie.max())

# 80/20 train/test split with a fixed seed
rng = np.random.default_rng(0)
is_test = rng.random(len(df)) < 0.2
train = df[~is_test]
test = df[is_test]
print("train / test sizes:", len(train), "/", len(test))

# plot 1: how many of each rating
counts = df["rating"].value_counts().sort_index()
plt.figure()
plt.bar(counts.index, counts.values)
plt.xlabel("rating (stars)")
plt.ylabel("count")
plt.title("Distribution of ratings")
plt.savefig("figures/rating_hist.png", dpi=200)

# plot 2: which entries of the matrix are observed
mask = np.zeros((n_users, n_movies), dtype=bool)
mask[df["user"], df["movie"]] = True
plt.figure()
plt.imshow(~mask, cmap="gray", aspect="auto")
plt.xlabel("movie")
plt.ylabel("user")
plt.title("Observed ratings (black)")
plt.savefig("figures/sparsity.png", dpi=200)

print("saved figures/rating_hist.png and figures/sparsity.png")
