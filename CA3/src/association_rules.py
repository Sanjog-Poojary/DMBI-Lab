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

import pandas as pd


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

