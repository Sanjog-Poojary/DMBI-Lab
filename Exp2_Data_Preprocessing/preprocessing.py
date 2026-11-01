import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
data_path = 'streaming_platform_dataset.csv' if os.path.exists('streaming_platform_dataset.csv') else 'dataset/streaming_platform_dataset.csv'
df = pd.read_csv(data_path)
print(f"Original Shape: {df.shape}")

# 2. Remove Duplicate Records
duplicates = df.duplicated().sum()
print(f"Duplicates Found: {duplicates}")
df = df.drop_duplicates()
print(f"Shape After Dropping Duplicates: {df.shape}")

# 3. Handle Missing Values
# Numerical imputation using mean
num_cols = ['age', 'watch_duration_min', 'completion_pct']
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical imputation using mode (most frequent value)
df['country'] = df['country'].fillna(df['country'].mode()[0])

# 4. Encode Categorical Data
cat_cols = ['subscription_tier', 'device_type', 'genre', 'country']
label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# 5. Feature Scaling (Standardization: Mean=0, Std=1)
scale_cols = ['age', 'watch_duration_min', 'completion_pct', 'monthly_spend']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Display Cleaned Output
print("\nFinal Preprocessed Data Head:")
print(df.head())
