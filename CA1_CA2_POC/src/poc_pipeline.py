"""
poc_pipeline.py — CA1 / CA2 analysis pipeline.
All functions are pure (no file I/O side effects) except when run as __main__.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "streaming_platform_dataset.csv")
_OUT  = os.path.join(_HERE, "..", "outputs")


# ── 1. Load and clean ────────────────────────────────────────────────────────
def load_and_clean(csv_path=None):
    """Return (raw_df, cleaned_df).

    Cleaning steps:
    1. Remove duplicate rows first (yields 502 rows).
    2. Strip / title-case subscription_tier.
    3. Fill missing numerics with column mean.
    4. Fill missing country with mode.
    5. Cap watch_duration_min to [0, 300] (removes sentinel 9999 etc.).
    """
    path = csv_path or _DATA
    raw_df = pd.read_csv(path)

    df = raw_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["subscription_tier"] = (
        df["subscription_tier"].astype(str).str.strip().str.title()
    )
    for col in ["age", "watch_duration_min", "completion_pct"]:
        df[col] = df[col].fillna(df[col].mean())
    df["country"] = df["country"].fillna(df["country"].mode()[0])
    df["watch_duration_min"] = df["watch_duration_min"].clip(lower=0, upper=300)

    return raw_df, df


# ── 2. Star schema ───────────────────────────────────────────────────────────
def build_star_schema(df):
    """Return (fact_df, dims_dict).

    Fact: Fact_Streams with surrogate FKs.
    Dims: Dim_User, Dim_Content (genre), Dim_Device.
    """
    df = df.copy().reset_index(drop=True)
    df["stream_id"] = df.index + 1

    dim_user = (
        df[["user_id", "age", "subscription_tier", "country", "monthly_spend"]]
        .drop_duplicates("user_id")
        .reset_index(drop=True)
        .copy()
    )
    dim_user.insert(0, "user_sk", range(1, len(dim_user) + 1))

    dim_content = (
        pd.DataFrame({"genre": sorted(df["genre"].unique())})
        .reset_index(drop=True)
    )
    dim_content.insert(0, "content_sk", range(1, len(dim_content) + 1))

    dim_device = (
        pd.DataFrame({"device_type": sorted(df["device_type"].unique())})
        .reset_index(drop=True)
    )
    dim_device.insert(0, "device_sk", range(1, len(dim_device) + 1))

    fact = (
        df
        .merge(dim_user[["user_sk", "user_id"]], on="user_id", how="left")
        .merge(dim_content, on="genre", how="left")
        .merge(dim_device, on="device_type", how="left")
    )

    fact_df = fact[
        ["stream_id", "user_sk", "content_sk", "device_sk",
         "watch_duration_min", "completion_pct", "monthly_spend"]
    ].copy()

    dims = {
        "Dim_User":    dim_user,
        "Dim_Content": dim_content,
        "Dim_Device":  dim_device,
    }
    return fact_df, dims


# ── 3. OLAP views ────────────────────────────────────────────────────────────
def olap_views(df):
    """Return (rollup_pivot, premium_genre_series, dice_series).

    rollup_pivot         — avg watch_duration_min  tier × genre  (roll-up).
    premium_genre_series — mean monthly_spend by genre for Premium  (slice).
    dice_series          — total monthly_spend  tier × country  (dice).
    """
    rollup_pivot = df.pivot_table(
        values="watch_duration_min",
        index="subscription_tier",
        columns="genre",
        aggfunc="mean",
    )

    premium = df[df["subscription_tier"] == "Premium"]
    premium_genre_series = (
        premium.groupby("genre")["monthly_spend"]
        .mean()
        .sort_values(ascending=False)
    )

    dice_series = df.groupby(
        ["subscription_tier", "country"]
    )["monthly_spend"].sum()

    return rollup_pivot, premium_genre_series, dice_series


# ── 4. Outlier detection ─────────────────────────────────────────────────────
def detect_outliers(raw_df):
    """Return dict with IQR and Isolation Forest results on raw watch_duration_min."""
    col = raw_df["watch_duration_min"].dropna()

    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    iqr_flags = col[(col < lower) | (col > upper)]

    X = col.values.reshape(-1, 1)
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X)
    iso_flags = col.iloc[preds == -1]

    return {
        "iqr_lower":  round(float(lower), 2),
        "iqr_upper":  round(float(upper), 2),
        "iqr_count":  int(len(iqr_flags)),
        "iqr_values": iqr_flags.values.tolist(),
        "iso_count":  int((preds == -1).sum()),
        "iso_values": iso_flags.values.tolist(),
    }


# ── 5. Subscriber segmentation ───────────────────────────────────────────────
def segment_subscribers(df):
    """Return dict with clustering results.

    Keys: silhouette_by_k, best_k, cluster_labels, pca_coords,
          cluster_sizes, dbscan_noise, cluster_profile.
    """
    features = ["age", "watch_duration_min", "completion_pct", "monthly_spend"]
    X_scaled = StandardScaler().fit_transform(df[features].copy())

    sil = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil[k] = round(float(silhouette_score(X_scaled, labels)), 4)

    best_k = max(sil, key=sil.get)
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_best.fit_predict(X_scaled)

    df2 = df.copy()
    df2["cluster"] = cluster_labels
    cluster_profile = df2.groupby("cluster")[features].mean().round(2)
    cluster_sizes   = df2["cluster"].value_counts().sort_index().to_dict()

    db_labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(X_scaled)
    dbscan_noise = int((db_labels == -1).sum())

    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    return {
        "silhouette_by_k":  sil,
        "best_k":           best_k,
        "cluster_labels":   cluster_labels.tolist(),
        "pca_coords":       coords.tolist(),
        "cluster_sizes":    {int(k): int(v) for k, v in cluster_sizes.items()},
        "dbscan_noise":     dbscan_noise,
        "cluster_profile":  cluster_profile.to_dict(),
    }


# ── 6. Churn classification ──────────────────────────────────────────────────
def classify_churn(df):
    """Return (results_dict, fitted_decision_tree, feature_names).

    Churn: completion_pct < 50  (binary label: Churned / Retained).
    Models: Decision Tree (balanced), Naive Bayes (balanced priors), Random Forest (balanced).
    """
    df2 = df.copy()
    df2["churn"] = (df2["completion_pct"] < 50).astype(int)

    le = LabelEncoder()
    for col in ["subscription_tier", "device_type", "genre", "country"]:
        df2[col] = le.fit_transform(df2[col].astype(str))

    feature_names = [
        "age", "subscription_tier", "device_type",
        "watch_duration_min", "genre", "country", "monthly_spend",
    ]
    X, y = df2[feature_names], df2["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42
        ),
        "Naive Bayes":   GaussianNB(priors=[0.5, 0.5]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    fitted_dt = None

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cm = confusion_matrix(y_te, y_pred)
        results[name] = {
            "accuracy":         round(float(accuracy_score(y_te, y_pred)), 4),
            "precision":        round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
            "f1":               round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
            "cv_accuracy":      round(float(cv_scores.mean()), 4),
            "confusion_matrix": cm.tolist(),
        }
        if name == "Decision Tree":
            fitted_dt = model

    return results, fitted_dt, feature_names


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(_OUT, exist_ok=True)

    raw_df, df = load_and_clean()
    print(f"Raw rows: {len(raw_df)}  Cleaned rows: {len(df)}")

    fact_df, dims = build_star_schema(df)
    print(f"Fact rows: {len(fact_df)}")

    olap_views(df)
    print("OLAP views built.")

    outliers = detect_outliers(raw_df)
    print(f"IQR outliers: {outliers['iqr_count']}  ISO outliers: {outliers['iso_count']}")

    seg = segment_subscribers(df)
    print(f"Best k: {seg['best_k']}  DBSCAN noise: {seg['dbscan_noise']}")

    churn_results, dt, fnames = classify_churn(df)
    print("Churn model results:")
    for m, r in churn_results.items():
        print(f"  {m}: acc={r['accuracy']}  recall={r['recall']}")

    out = {
        "raw_rows":     len(raw_df),
        "cleaned_rows": len(df),
        "fact_rows":    len(fact_df),
        "outliers":     outliers,
        "segmentation": {
            k: v for k, v in seg.items()
            if k not in ("cluster_labels", "pca_coords")
        },
        "churn": churn_results,
    }
    out_path = os.path.join(_OUT, "poc_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
