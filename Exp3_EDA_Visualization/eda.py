import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Load dataset
data_path = 'streaming_platform_dataset.csv' if os.path.exists('streaming_platform_dataset.csv') else 'dataset/streaming_platform_dataset.csv'
df = pd.read_csv(data_path)

print("--- Dataset Information ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# ---- Task 1: Central tendency ----
num_cols = ['age', 'watch_duration_min', 'completion_pct', 'monthly_spend']
cat_cols = ['subscription_tier', 'device_type', 'genre', 'country']

central_tendency = {
    'Mean': df[num_cols].mean(),
    'Median': df[num_cols].median(),
    'Mode': df[cat_cols].mode().iloc[0],
}
print("=== Central Tendency Measures ===")
for measure, values in central_tendency.items():
    print(f"\n{measure}:")
    print(values)

# ---- Task 2: Dispersion ----
dispersion = {
    'Standard Deviation': df[num_cols].std(),
    'Variance': df[num_cols].var(),
    'Range': df[num_cols].max() - df[num_cols].min(),
    'IQR': df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25),
}
print("=== Dispersion Measures ===")
for measure, values in dispersion.items():
    print(f"\n{measure}:")
    print(values)

# ---- Task 3: Correlation ----
correlation_matrix = df[num_cols].corr()
print("=== Correlation Matrix ===")
print(correlation_matrix.round(4))

# ---- Task 4: Visualizations ----
# 4a. Histogram of watch duration
plt.figure(figsize=(10, 5))
if sns is not None:
    sns.histplot(data=df, x='watch_duration_min', bins=30, kde=True, color='#2b5c8f')
else:
    plt.hist(df['watch_duration_min'].dropna(), bins=30, color='#2b5c8f', edgecolor='black')
plt.title('Distribution of Watch Duration (Minutes)', fontsize=14)
plt.xlabel('Watch Duration (min)')
plt.ylabel('User Count')
plt.show()

# 4b. Boxplot before cleanup (inconsistent casing in subscription_tier)
plt.figure(figsize=(10, 5))
if sns is not None:
    sns.boxplot(data=df, x='subscription_tier', y='monthly_spend', palette='Blues_r')
else:
    tiers = df['subscription_tier'].unique()
    data = [df[df['subscription_tier'] == t]['monthly_spend'].dropna() for t in tiers]
    plt.boxplot(data, tick_labels=tiers)
plt.title('Boxplot of Monthly Spend by Subscription Tier', fontsize=14)
plt.xlabel('Subscription Tier')
plt.ylabel('Monthly Spend')
plt.xticks(rotation=45)
plt.show()

# 4c. Boxplot after cleanup
df['subscription_tier'] = df['subscription_tier'].astype(str).str.strip().str.title()
plt.figure(figsize=(8, 5))
if sns is not None:
    sns.boxplot(data=df, x='subscription_tier', y='monthly_spend',
                hue='subscription_tier', palette='Blues_r')
else:
    tiers = df['subscription_tier'].unique()
    data = [df[df['subscription_tier'] == t]['monthly_spend'].dropna() for t in tiers]
    plt.boxplot(data, tick_labels=tiers)
plt.title('Boxplot of Monthly Spend by Subscription Tier', fontsize=14)
plt.xlabel('Subscription Tier')
plt.ylabel('Monthly Spend')
plt.tight_layout()
plt.show()

# 4d. Correlation heatmap
plt.figure(figsize=(8, 6))
if sns is not None:
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f', cbar=True)
else:
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=45)
    plt.yticks(range(len(num_cols)), num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.3f}", ha="center", va="center", color="black")
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.show()

# 4e. Scatter: completion % vs watch duration
plt.figure(figsize=(10, 6))
if sns is not None:
    sns.scatterplot(data=df, x='completion_pct', y='watch_duration_min', hue='genre', alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    plt.scatter(df['completion_pct'], df['watch_duration_min'], alpha=0.8, c='#2b5c8f')
plt.title('Scatter Plot: Completion % vs Watch Duration', fontsize=14)
plt.xlabel('Completion Percentage (%)')
plt.ylabel('Watch Duration (Minutes)')
plt.show()
