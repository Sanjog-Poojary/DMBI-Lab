"""Experiment 8: Apriori association mining on JioHotstar behaviour items (Python).

Put the dataset CSV in this folder as JioHotstar_Preprocessed_Dataset.csv and run:
    pip install mlxtend
    python apriori_association.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JioHotstar_Preprocessed_Dataset.csv")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Create behaviour-based items for Association Mining

transaction_df = pd.DataFrame(index=df.index)

transaction_df["High_Watch"] = df["avg_watch_percentage"] >= 70
transaction_df["Strong_Hook"] = df["hook_strength"] >= 7
transaction_df["High_Pacing"] = df["pacing_score"] >= 7
transaction_df["High_Visual_Intensity"] = df["visual_intensity"] >= 7
transaction_df["Skipped_Intro"] = df["skip_intro"] == 1
transaction_df["High_Cognitive_Load"] = df["cognitive_load"] >= 7
transaction_df["Night_Watch_Safe"] = df["night_watch_safe"] == 1

print("Transaction items created successfully!")
print("\nShape:", transaction_df.shape)
print("\nFirst 5 transactions:")
print(transaction_df.head())

# Convert True/False values into 1/0 transaction format

transaction_matrix = transaction_df.astype(int)

print("Transaction matrix created successfully!")
print("\nShape:", transaction_matrix.shape)
print("\nFirst 5 transactions:")
print(transaction_matrix.head())

# Check the frequency of each behaviour item

item_counts = transaction_matrix.sum().sort_values(ascending=False)

item_percentage = (item_counts / len(transaction_matrix) * 100).round(2)

frequency_table = pd.DataFrame({
    "Item_Count": item_counts,
    "Support_Percentage": item_percentage
})

print("Item frequency:")
print(frequency_table)

from mlxtend.frequent_patterns import apriori, association_rules

# Find frequent itemsets using Apriori
frequent_itemsets = apriori(
    transaction_matrix,
    min_support=0.05,
    use_colnames=True
)

print("Frequent itemsets generated successfully!")
print("Number of Frequent Itemsets:", len(frequent_itemsets))

print(frequent_itemsets.sort_values(by="support", ascending=False))

# Generate association rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.30
)

# Keep only positive associations
rules = rules[rules["lift"] >= 1]

# Sort by confidence and then lift
rules = rules.sort_values(
    by=["confidence", "lift"],
    ascending=False
)

print("Association rules generated successfully!")
print("Number of rules:", len(rules))

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# Format association rules for easy interpretation

top_rules = rules.copy()

top_rules["Antecedent"] = top_rules["antecedents"].apply(
    lambda x: ", ".join(sorted(x))
)

top_rules["Consequent"] = top_rules["consequents"].apply(
    lambda x: ", ".join(sorted(x))
)

final_rules = top_rules[
    ["Antecedent", "Consequent", "support", "confidence", "lift"]
].head(10)

final_rules = final_rules.round({
    "support": 4,
    "confidence": 4,
    "lift": 4
})

print("Top 10 Association Rules:")
print(final_rules)

import matplotlib.pyplot as plt

# Create rule labels
final_rules["Rule"] = (
    final_rules["Antecedent"] + " → " + final_rules["Consequent"]
)

# Plot confidence of top 10 rules
plt.figure(figsize=(10, 6))

plt.barh(
    final_rules["Rule"],
    final_rules["confidence"]
)

plt.xlabel("Confidence")
plt.ylabel("Association Rule")
plt.title("Top 10 Association Rules by Confidence")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Find the strongest association rule based on lift

strongest_rule = rules.loc[rules["lift"].idxmax()]

print("Strongest Association Rule:")
print("Antecedent :", strongest_rule["antecedents"])
print("Consequent :", strongest_rule["consequents"])
print("Support    :", round(strongest_rule["support"], 4))
print("Confidence :", round(strongest_rule["confidence"], 4))
print("Lift       :", round(strongest_rule["lift"], 4))

print("========== EXPERIMENT 8: APRIORI ASSOCIATION MINING ==========")

print("\nDataset:")
print("Total transactions:", len(transaction_matrix))
print("Total behaviour items:", len(transaction_matrix.columns))

print("\nApriori Parameters:")
print("Minimum Support:", 0.05)
print("Minimum Confidence:", 0.30)
print("Minimum Lift:", 1.0)

print("\nResults:")
print("Frequent Itemsets:", len(frequent_itemsets))
print("Positive Association Rules:", len(rules))

print("\nPython implementation completed successfully.")
