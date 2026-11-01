"""
generate_viewing_sessions.py
Generates CA3/data/viewing_sessions.csv — synthetic user-genre viewing sessions.

Each row: user_id, genres_watched (comma-separated list of genres per session).
500 users, 3–8 sessions each, genre co-viewing patterns planted.
"""

import os
import random

import pandas as pd

random.seed(42)

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance",
          "Sci-Fi", "Thriller", "Documentary"]

# Planted co-viewing patterns (pairs that tend to appear together)
CO_PAIRS = [
    ("Action", "Thriller"),
    ("Comedy", "Romance"),
    ("Drama", "Documentary"),
    ("Horror", "Thriller"),
    ("Sci-Fi", "Action"),
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT  = os.path.join(_HERE, "..", "data", "viewing_sessions.csv")


def generate(n_users=500, seed=42):
    random.seed(seed)
    rows = []
    for uid in range(1001, 1001 + n_users):
        n_sessions = random.randint(3, 8)
        for _ in range(n_sessions):
            # Pick a base genre
            base = random.choice(GENRES)
            session = {base}
            # With 60% probability add a co-genre
            for g1, g2 in CO_PAIRS:
                if base == g1 and random.random() < 0.6:
                    session.add(g2)
                elif base == g2 and random.random() < 0.6:
                    session.add(g1)
            # Occasionally add a random extra genre
            if random.random() < 0.2:
                session.add(random.choice(GENRES))
            rows.append({"user_id": uid, "genres_watched": ",".join(sorted(session))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    df = generate()
    df.to_csv(_OUT, index=False)
    print(f"Written {len(df)} sessions for {df['user_id'].nunique()} users -> {_OUT}")

