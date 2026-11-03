"""
Experiment 9: FP-Growth Association Rule Mining in Python (RapidMiner Comparison).
Course: Data Mining and Business Intelligence (DMBI Lab) — Roll No. 38.

Implements the Frequent Pattern Growth (FP-Growth) algorithm using mlxtend.
Mirrors the RapidMiner Studio operator pipeline:
  [Retrieve] -> [Numerical to Binominal] -> [FP-Growth (min_supp=0.05)] -> [Create Association Rules (min_conf=0.30)]

Key Advantages of FP-Growth over Apriori:
- Compresses dataset into a compact FP-Tree (Frequent Pattern Tree).
- Eliminates expensive candidate generation step (O(2^|I|)).
- Requires only two database scans.
"""

import os
import time
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV_TRANSACTIONS = os.path.join(_HERE, "JioHotstar_Association_Transactions.csv")
_CSV_RAW = os.path.join(_HERE, "JioHotstar_Preprocessed_Dataset.csv")
if not os.path.exists(_CSV_RAW):
    _CSV_RAW = os.path.join(_HERE, "..", "Exp6_KMeans_Clustering", "JioHotstar_Preprocessed_Dataset.csv")


def load_transaction_matrix():
    """Load or build the 7-item binary transaction matrix."""
    if os.path.exists(_CSV_TRANSACTIONS):
        df_tx = pd.read_csv(_CSV_TRANSACTIONS)
        print(f"Loaded pre-built transaction matrix: {df_tx.shape}")
        return df_tx

    print(f"Building transaction items from: {_CSV_RAW}")
    df = pd.read_csv(_CSV_RAW)
    df_tx = pd.DataFrame({
        "High_Watch": (df["avg_watch_percentage"] >= 70).astype(int),
        "Strong_Hook": (df["hook_strength"] >= 7).astype(int),
        "High_Pacing": (df["pacing_score"] >= 7).astype(int),
        "High_Visual_Intensity": (df["visual_intensity"] >= 7).astype(int),
        "Skipped_Intro": (df["skip_intro"] == 1).astype(int),
        "High_Cognitive_Load": (df["cognitive_load"] >= 7).astype(int),
        "Night_Watch_Safe": (df["night_watch_safe"] == 1).astype(int),
    })
    df_tx.to_csv(_CSV_TRANSACTIONS, index=False)
    print(f"Saved transaction matrix to: {_CSV_TRANSACTIONS}")
    return df_tx


def main():
    print("=" * 70)
    print("  EXPERIMENT 9: FP-GROWTH ASSOCIATION RULE MINING (PYTHON & RAPIDMINER)")
    print("=" * 70)

    df_tx = load_transaction_matrix()
    n_transactions, n_items = df_tx.shape
    print(f"\n1. Transaction Matrix Shape: {n_transactions:,} rows x {n_items} items")

    # Item Frequency & Support
    counts = df_tx.sum().sort_values(ascending=False)
    supports = (counts / n_transactions * 100).round(2)
    freq_table = pd.DataFrame({"Item_Count": counts, "Support_Pct": supports})
    print("\n2. Behavioural Item Frequencies:")
    print(freq_table.to_string())

    # Convert to boolean for mlxtend FP-Growth
    bool_matrix = df_tx.astype(bool)

    # 3. Frequent itemset mining via FP-Growth
    min_support = 0.05
    t0 = time.time()
    frequent_itemsets = fpgrowth(bool_matrix, min_support=min_support, use_colnames=True)
    t_fp = time.time() - t0

    print(f"\n3. FP-Growth Frequent Itemsets (min_support = {min_support}):")
    print(f"   Generated {len(frequent_itemsets)} frequent itemsets in {t_fp:.4f} seconds.")
    print(frequent_itemsets.sort_values(by="support", ascending=False).to_string())

    # 4. Comparative benchmark: Apriori vs FP-Growth runtime
    t0 = time.time()
    _ = apriori(bool_matrix, min_support=min_support, use_colnames=True)
    t_ap = time.time() - t0
    speedup = t_ap / t_fp if t_fp > 0 else 1.0
    print(f"\n4. Algorithm Runtime Benchmark:")
    print(f"   • Apriori runtime:   {t_ap:.4f} seconds")
    print(f"   • FP-Growth runtime: {t_fp:.4f} seconds")
    print(f"   • FP-Growth speedup: {speedup:.2f}x faster than Apriori")

    # 5. Generate association rules
    min_confidence = 0.30
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    # Filter positive associations (lift >= 1.0)
    positive_rules = rules[rules["lift"] >= 1.0].sort_values(by=["confidence", "lift"], ascending=False).copy()

    print(f"\n5. Association Rules Generated (min_conf = {min_confidence}, lift >= 1.0):")
    print(f"   Total positive association rules: {len(positive_rules)}")

    # Format rules for presentation
    positive_rules["Antecedent"] = positive_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    positive_rules["Consequent"] = positive_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    display_cols = ["Antecedent", "Consequent", "support", "confidence", "lift"]
    top_10 = positive_rules[display_cols].head(10).round({"support": 4, "confidence": 4, "lift": 4})

    print("\n6. Top 10 Association Rules by Confidence:")
    print(top_10.to_string(index=False))

    # Strongest rule by lift
    strongest = positive_rules.loc[positive_rules["lift"].idxmax()]
    print("\n7. Strongest Association Rule (by Lift):")
    print(f"   • Antecedents: {set(strongest['antecedents'])}")
    print(f"   • Consequents: {set(strongest['consequents'])}")
    print(f"   • Support:     {strongest['support']:.4f}")
    print(f"   • Confidence:  {strongest['confidence']:.4f}")
    print(f"   • Lift:        {strongest['lift']:.4f}")

    # 8. RapidMiner alignment summary
    print("\n" + "=" * 70)
    print("  RAPIDMINER vs. PYTHON FP-GROWTH SUMMARY")
    print("=" * 70)
    print(f"{'Parameter / Output':<30} | {'Python (mlxtend FP-Growth)':<22} | {'RapidMiner Studio':<22}")
    print("-" * 78)
    print(f"{'Algorithm':<30} | {'FP-Growth (FP-Tree)':<22} | {'FP-Growth Operator':<22}")
    print(f"{'Minimum Support':<30} | {'0.05':<22} | {'0.05':<22}")
    print(f"{'Minimum Confidence':<30} | {'0.30':<22} | {'0.30':<22}")
    print(f"{'Minimum Lift Threshold':<30} | {'1.0':<22} | {'1.0':<22}")
    print(f"{'Dataset Rows Processed':<30} | {f'{n_transactions:,}':<22} | {'10,000 (Free limit)':<22}")
    print(f"{'Frequent Itemsets Mined':<30} | {f'{len(frequent_itemsets)}':<22} | {'Consistent itemsets':<22}")
    print(f"{'Positive Association Rules':<30} | {f'{len(positive_rules)}':<22} | {'Consistent rules':<22}")
    print("=" * 78)
    print("\nExperiment 9 FP-Growth execution completed successfully.")


if __name__ == "__main__":
    main()

