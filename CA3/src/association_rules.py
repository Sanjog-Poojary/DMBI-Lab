"""
association_rules.py — from-scratch Apriori for CA3.

Public API:
  build_baskets(sessions_df)         -> list[frozenset]
  apriori(baskets, min_support)      -> dict[frozenset, float]
  association_rules(frequent,        -> pd.DataFrame
                    min_confidence,
                    min_lift)
  recommend(user_genres_set,         -> list[str]
            rules_df, top_n)
"""

from itertools import combinations
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "viewing_sessions.csv")
if not os.path.exists(_DATA):
    _DATA = os.path.abspath(os.path.join(_HERE, "..", "..", "CA3", "data", "viewing_sessions.csv"))
_OUT_DIR = os.path.join(_HERE, "..", "outputs")
_OUT_CSV = os.path.join(_OUT_DIR, "association_rules.csv")


# ── helpers ──────────────────────────────────────────────────────────────────
def _support(itemset, baskets):
    return sum(1 for b in baskets if itemset.issubset(b)) / len(baskets)


def _generate_candidates(prev_freq, k):
    items = sorted({item for fs in prev_freq for item in fs})
    return [frozenset(c) for c in combinations(items, k)]


# ── public functions ──────────────────────────────────────────────────────────
def build_baskets(sessions_df):
    """Convert sessions DataFrame to list of frozensets.

    Expects column 'genres_watched' with comma-separated genre strings.
    """
    baskets = []
    for row in sessions_df["genres_watched"]:
        genres = [g.strip() for g in str(row).split(",") if g.strip()]
        if genres:
            baskets.append(frozenset(genres))
    return baskets


def apriori(baskets, min_support=0.05):
    """Return dict mapping frequent frozensets to their support values."""
    n = len(baskets)
    if n == 0:
        return {}

    # 1-itemsets
    all_items = sorted({item for b in baskets for item in b})
    frequent = {}
    current = []
    for item in all_items:
        sup = sum(1 for b in baskets if item in b) / n
        if sup >= min_support:
            fs = frozenset([item])
            frequent[fs] = round(sup, 6)
            current.append(fs)

    # k-itemsets
    k = 2
    while current:
        candidates = _generate_candidates(current, k)
        current = []
        for cand in candidates:
            sup = _support(cand, baskets)
            if sup >= min_support:
                frequent[cand] = round(sup, 6)
                current.append(cand)
        k += 1

    return frequent


def association_rules(frequent, min_confidence=0.4, min_lift=1.1):
    """Derive association rules from frequent itemsets.

    Returns a DataFrame with columns:
        antecedent, consequent, support, confidence, lift.
    """
    records = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for ant in combinations(sorted(itemset), size):
                ant_fs  = frozenset(ant)
                cons_fs = itemset - ant_fs
                ant_sup = frequent.get(ant_fs)
                if ant_sup is None or ant_sup == 0:
                    continue
                conf = sup / ant_sup
                if conf < min_confidence:
                    continue
                cons_sup = frequent.get(cons_fs)
                if cons_sup is None or cons_sup == 0:
                    continue
                lift = conf / cons_sup
                if lift < min_lift:
                    continue
                records.append({
                    "antecedent":  ", ".join(sorted(ant_fs)),
                    "consequent":  ", ".join(sorted(cons_fs)),
                    "support":     round(sup, 4),
                    "confidence":  round(conf, 4),
                    "lift":        round(lift, 4),
                })

    if not records:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])

    return (
        pd.DataFrame(records)
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset=["antecedent", "consequent"])
        .reset_index(drop=True)
    )


def recommend(user_genres_set, rules_df, top_n=5):
    """Return up to top_n genre recommendations for a user.

    Matches rules whose antecedent genres are all in user_genres_set,
    picks the consequent genres not already watched, ranked by lift.
    """
    if rules_df.empty:
        return []

    recs = {}
    for _, row in rules_df.iterrows():
        ant = set(row["antecedent"].split(", "))
        cons = set(row["consequent"].split(", "))
        if ant.issubset(user_genres_set):
            for genre in cons - user_genres_set:
                if genre not in recs or recs[genre] < row["lift"]:
                    recs[genre] = row["lift"]

    return sorted(recs, key=recs.get, reverse=True)[:top_n]


def main():
    if not os.path.exists(_DATA):
        print(f"Dataset not found at {_DATA}")
        return

    sessions_df = pd.read_csv(_DATA)
    baskets = build_baskets(sessions_df)
    n_users = sessions_df["user_id"].nunique()

    frequent = apriori(baskets, min_support=0.05)
    rules_df = association_rules(frequent, min_confidence=0.4, min_lift=1.1)

    print(f"Number of users: {n_users}")
    print(f"Frequent itemsets: {len(frequent)}")
    print(f"Association rules: {len(rules_df)}")
    print("\n--- Top 10 Association Rules by Lift ---")
    print(rules_df.head(10).to_string(index=False))

    os.makedirs(_OUT_DIR, exist_ok=True)
    rules_df.to_csv(_OUT_CSV, index=False)
    print(f"\nRules saved to: {_OUT_CSV}")

    # Sample user recommendations
    sample_uid = None
    sample_recs = []
    sample_genres = set()
    for uid in sessions_df["user_id"].unique():
        ug = set(
            sessions_df[sessions_df["user_id"] == uid]["genres_watched"]
            .str.split(",")
            .explode()
            .str.strip()
            .unique()
        )
        r = recommend(ug, rules_df, top_n=5)
        if r:
            sample_uid = uid
            sample_genres = ug
            sample_recs = r
            break
    if sample_uid is None:
        sample_uid = sessions_df["user_id"].iloc[0]
        sample_genres = set(
            sessions_df[sessions_df["user_id"] == sample_uid]["genres_watched"]
            .str.split(",")
            .explode()
            .str.strip()
            .unique()
        )
        sample_recs = recommend(sample_genres, rules_df, top_n=5)

    print(f"\nSample user {sample_uid} watched: {sorted(sample_genres)}")
    print(f"Recommended genres: {sample_recs}")


if __name__ == "__main__":
    main()
