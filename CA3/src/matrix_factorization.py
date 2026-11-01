"""
matrix_factorization.py — Collaborative Filtering with Matrix Factorization for CA3.
Topic: Collaborative Filtering Recommender Systems with Matrix Factorization for Streaming Platforms (Beyond Syllabus).

Compares:
1. Popularity Baseline (Most frequently watched genres)
2. Association Rules Baseline (Apriori rule-based recommender from CA2)
3. Low-Rank Matrix Factorization (ALS on user-genre interaction matrix)

Metric:
- Hit Rate @ 3 on held-out test interactions and test users.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Path resolution ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Ensure CA2_Mini_Project/src and CA3/src are on sys.path to import association_rules
for _p in (os.path.join(_REPO, "CA2_Mini_Project", "src"),
           os.path.join(_REPO, "CA3", "src"),
           _HERE):
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from association_rules import apriori, association_rules, recommend
except ImportError:
    # Fallback inline dummy if association_rules cannot be located
    apriori = None
    association_rules = None
    recommend = None

_DATA_FILE = os.path.join(_HERE, "..", "data", "viewing_sessions.csv")
if not os.path.exists(_DATA_FILE):
    _DATA_FILE = os.path.join(_REPO, "CA3", "data", "viewing_sessions.csv")

_OUT_DIR = os.path.join(_HERE, "..", "outputs")
_OUT_JSON = os.path.join(_OUT_DIR, "cf_results.json")


# ── Data extraction ───────────────────────────────────────────────────────────
def load_and_build_user_genres(csv_path):
    """Load viewing sessions and extract set of genres watched by each user."""
    df = pd.read_csv(csv_path)
    user_genres = (
        df.groupby("user_id")["genres_watched"]
        .apply(lambda s: sorted(set(g.strip() for r in s for g in str(r).split(",") if g.strip())))
        .to_dict()
    )
    all_genres = sorted(set(g for g_list in user_genres.values() for g in g_list))
    all_users = sorted(user_genres.keys())
    return user_genres, all_users, all_genres


def create_train_test_split(user_genres, test_size=0.2, random_state=42):
    """Split observed (user, genre) pairs into train and test sets."""
    pairs = [(u, g) for u, g_list in user_genres.items() for g in g_list]
    train_pairs, test_pairs = train_test_split(pairs, test_size=test_size, random_state=random_state)
    return train_pairs, test_pairs


# ── Matrix Factorization (ALS) ────────────────────────────────────────────────
class MatrixFactorizationALS:
    """Explicit/implicit matrix factorization solved via Alternating Least Squares (ALS).

    Learns user latent matrix U (N x k) and item latent matrix V (M x k) such that
    R_hat = U @ V.T approximates the binary user-genre interaction matrix R.
    """

    def __init__(self, rank=8, reg=0.1, n_iters=20, random_state=42):
        self.rank = rank
        self.reg = reg
        self.n_iters = n_iters
        self.random_state = random_state
        self.U = None
        self.V = None

    def fit(self, R):
        """Fit ALS model on user-item matrix R (shape: N x M)."""
        rng = np.random.RandomState(self.random_state)
        n_users, n_items = R.shape

        self.U = rng.normal(0, 0.1, (n_users, self.rank)).astype(np.float64)
        self.V = rng.normal(0, 0.1, (n_items, self.rank)).astype(np.float64)

        reg_I = self.reg * np.eye(self.rank, dtype=np.float64)

        for _ in range(self.n_iters):
            # Update user factors U: U = (R @ V) @ (V.T @ V + reg*I)^(-1)
            vtv = self.V.T @ self.V + reg_I
            self.U = (R @ self.V) @ np.linalg.inv(vtv)

            # Update item factors V: V = (R.T @ U) @ (U.T @ U + reg*I)^(-1)
            utu = self.U.T @ self.U + reg_I
            self.V = (R.T @ self.U) @ np.linalg.inv(utu)

        return self

    def predict_matrix(self):
        """Compute full predicted score matrix R_pred = U @ V.T."""
        return self.U @ self.V.T


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_recommenders(user_genres, train_pairs, test_pairs, all_users, all_genres, rank=8, reg=0.1, top_k=3, random_state=42):
    """Train Popularity, Association Rules, and Matrix Factorization models,

    and evaluate their Hit Rate @ top_k on the test set.
    """
    user2idx = {u: i for i, u in enumerate(all_users)}
    genre2idx = {g: i for i, g in enumerate(all_genres)}
    n_users = len(all_users)
    n_genres = len(all_genres)

    # 1. Build training interaction matrix and training baskets
    R_train = np.zeros((n_users, n_genres), dtype=np.float64)
    train_user_genres = {u: set() for u in all_users}
    for u, g in train_pairs:
        R_train[user2idx[u], genre2idx[g]] = 1.0
        train_user_genres[u].add(g)

    # Test genres per user
    test_user_genres = {}
    for u, g in test_pairs:
        test_user_genres.setdefault(u, set()).add(g)

    # ── Model 1: Popularity baseline ──────────────────────────────────────────
    genre_pop_counts = R_train.sum(axis=0)

    # ── Model 2: Association rules baseline ───────────────────────────────────
    train_baskets = [frozenset(g_set) for g_set in train_user_genres.values() if g_set]
    if apriori and association_rules:
        frequent = apriori(train_baskets, min_support=0.05)
        rules_df = association_rules(frequent, min_confidence=0.4, min_lift=1.1)
    else:
        rules_df = pd.DataFrame()

    # ── Model 3: Matrix Factorization (ALS) ───────────────────────────────────
    mf_model = MatrixFactorizationALS(rank=rank, reg=reg, n_iters=25, random_state=random_state)
    mf_model.fit(R_train)
    R_pred = mf_model.predict_matrix()

    # ── Evaluation: User-level and Interaction-level Hit Rate @ top_k ─────────
    # Evaluated users: test users who have at least one unseen candidate genre
    user_hits_pop = 0
    user_hits_ar = 0
    user_hits_mf = 0
    n_eval_users = 0

    for u, test_g_set in test_user_genres.items():
        candidates = [g for g in all_genres if g not in train_user_genres[u]]
        if not candidates:
            continue
        n_eval_users += 1

        # 1. Pop recs
        pop_recs = sorted(candidates, key=lambda g: genre_pop_counts[genre2idx[g]], reverse=True)[:top_k]
        if any(g in test_g_set for g in pop_recs):
            user_hits_pop += 1

        # 2. AR recs
        if recommend and not rules_df.empty:
            ar_recs = recommend(train_user_genres[u], rules_df, top_n=top_k)
        else:
            ar_recs = []
        if any(g in test_g_set for g in ar_recs):
            user_hits_ar += 1

        # 3. MF recs
        u_idx = user2idx[u]
        mf_recs = sorted(candidates, key=lambda g: R_pred[u_idx, genre2idx[g]], reverse=True)[:top_k]
        if any(g in test_g_set for g in mf_recs):
            user_hits_mf += 1

    # Interaction-level hit rate (over all test pairs)
    pair_hits_pop = 0
    pair_hits_ar = 0
    pair_hits_mf = 0
    n_eval_pairs = len(test_pairs)

    for u, g in test_pairs:
        candidates = [item for item in all_genres if item not in train_user_genres[u]]
        u_idx = user2idx[u]

        pop_recs = sorted(candidates, key=lambda item: genre_pop_counts[genre2idx[item]], reverse=True)[:top_k]
        if recommend and not rules_df.empty:
            ar_recs = recommend(train_user_genres[u], rules_df, top_n=top_k)
        else:
            ar_recs = []
        mf_recs = sorted(candidates, key=lambda item: R_pred[u_idx, genre2idx[item]], reverse=True)[:top_k]

        if g in pop_recs:
            pair_hits_pop += 1
        if g in ar_recs:
            pair_hits_ar += 1
        if g in mf_recs:
            pair_hits_mf += 1

    results = {
        "dataset_summary": {
            "total_users": n_users,
            "total_genres": n_genres,
            "total_observed_interactions": len(train_pairs) + len(test_pairs),
            "train_interactions": len(train_pairs),
            "test_interactions": len(test_pairs),
            "evaluated_test_users": n_eval_users,
            "top_k": top_k,
            "rank": rank,
            "regularization": reg,
            "random_state": random_state,
        },
        "hit_rate_user_level": {
            "Popularity": round(user_hits_pop / n_eval_users, 4) if n_eval_users else 0.0,
            "Association Rules": round(user_hits_ar / n_eval_users, 4) if n_eval_users else 0.0,
            "Matrix Factorization": round(user_hits_mf / n_eval_users, 4) if n_eval_users else 0.0,
        },
        "hit_rate_interaction_level": {
            "Popularity": round(pair_hits_pop / n_eval_pairs, 4) if n_eval_pairs else 0.0,
            "Association Rules": round(pair_hits_ar / n_eval_pairs, 4) if n_eval_pairs else 0.0,
            "Matrix Factorization": round(pair_hits_mf / n_eval_pairs, 4) if n_eval_pairs else 0.0,
        },
        "user_hit_counts": {
            "Popularity": f"{user_hits_pop}/{n_eval_users}",
            "Association Rules": f"{user_hits_ar}/{n_eval_users}",
            "Matrix Factorization": f"{user_hits_mf}/{n_eval_users}",
        },
        "interaction_hit_counts": {
            "Popularity": f"{pair_hits_pop}/{n_eval_pairs}",
            "Association Rules": f"{pair_hits_ar}/{n_eval_pairs}",
            "Matrix Factorization": f"{pair_hits_mf}/{n_eval_pairs}",
        },
    }

    return results, mf_model, rules_df


def main():
    print("=" * 70)
    print("  CA3 Content Beyond Syllabus: Matrix Factorization Recommender")
    print("=" * 70)

    if not os.path.exists(_DATA_FILE):
        print(f"Error: viewing sessions data not found at {_DATA_FILE}")
        return

    print(f"Loading session data from: {_DATA_FILE}")
    user_genres, all_users, all_genres = load_and_build_user_genres(_DATA_FILE)
    print(f"Loaded {len(all_users)} unique users across {len(all_genres)} genres: {all_genres}")

    train_pairs, test_pairs = create_train_test_split(user_genres, test_size=0.2, random_state=42)
    print(f"Train/Test split (80/20): {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

    print("\nTraining models and evaluating Hit Rate @ 3...")
    results, mf_model, rules_df = evaluate_recommenders(
        user_genres, train_pairs, test_pairs, all_users, all_genres, rank=8, reg=0.1, top_k=3, random_state=42
    )

    print("\n" + "-" * 70)
    print(f"{'Method':<25} | {'User Hit Rate @ 3':<20} | {'Interaction Hit Rate @ 3':<20}")
    print("-" * 70)
    for method in ["Popularity", "Association Rules", "Matrix Factorization"]:
        u_hr = results["hit_rate_user_level"][method]
        u_cnt = results["user_hit_counts"][method]
        i_hr = results["hit_rate_interaction_level"][method]
        i_cnt = results["interaction_hit_counts"][method]
        print(f"{method:<25} | {u_hr:.4f} ({u_cnt})        | {i_hr:.4f} ({i_cnt})")
    print("-" * 70)

    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {_OUT_JSON}")


if __name__ == "__main__":
    main()
