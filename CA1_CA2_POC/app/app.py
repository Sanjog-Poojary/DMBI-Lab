"""
app.py — DMBI Streaming Platform BI Dashboard.

Run from repo root:
    streamlit run CA1_CA2_POC/app/app.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.abspath(os.path.join(_APP_DIR, "..", ".."))
for _p in (os.path.join(_REPO, "CA1_CA2_POC", "src"),
           os.path.join(_REPO, "CA3", "src"),
           _APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers import show_banner, kpi_row, make_confusion_heatmap, apply_sidebar_filters
import poc_pipeline as pp
from association_rules import build_baskets, apriori, association_rules, recommend

# ── Paths to data ──────────────────────────────────────────────────────────────
_CSV_MAIN = os.path.join(_REPO, "CA1_CA2_POC", "data", "streaming_platform_dataset.csv")
_CSV_CA3  = os.path.join(_REPO, "CA3", "data", "viewing_sessions.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DMBI Streaming BI", layout="wide")

# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pp.load_and_clean(_CSV_MAIN)


@st.cache_data
def get_schema(cleaned_df):
    return pp.build_star_schema(cleaned_df)


@st.cache_data
def get_olap(cleaned_df):
    return pp.olap_views(cleaned_df)


@st.cache_data
def get_outliers(raw_df):
    return pp.detect_outliers(raw_df)


@st.cache_data
def get_segments(cleaned_df):
    return pp.segment_subscribers(cleaned_df)


@st.cache_data
def get_churn(cleaned_df):
    return pp.classify_churn(cleaned_df)


@st.cache_data
def get_rules(min_support, min_confidence, min_lift):
    sessions = pd.read_csv(_CSV_CA3)
    baskets  = build_baskets(sessions)
    frequent = apriori(baskets, min_support=min_support)
    rules    = association_rules(frequent, min_confidence=min_confidence, min_lift=min_lift)
    return sessions, baskets, rules


# ── Sidebar ───────────────────────────────────────────────────────────────────
raw_df, cleaned_df = load_data()

st.sidebar.title("Filters")
df_filtered = apply_sidebar_filters(cleaned_df)

page = st.sidebar.radio(
    "Page",
    ["Overview", "Warehouse & OLAP", "Outliers", "Segments", "Churn model", "Recommendations"],
)

# ── Page 1: Overview ──────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Overview  (Module II — Data Preprocessing)")
    show_banner()

    fact_df, _ = get_schema(cleaned_df)
    mean_watch = df_filtered["watch_duration_min"].mean()
    mean_watch_str = f"{mean_watch:.1f}" if not np.isnan(mean_watch) else "0.0"
    kpi_row([
        ("Raw rows",            f"{len(raw_df):,}"),
        ("After de-dup",        f"{len(cleaned_df):,}"),
        ("Fact rows",           f"{len(fact_df):,}"),
        ("Total monthly spend", f"${df_filtered['monthly_spend'].sum():,.0f}"),
        ("Mean watch (min)",    mean_watch_str),
    ])

    st.subheader("Cleaning summary")
    raw_missing = raw_df.isnull().sum().rename("Missing (raw)")
    clean_missing = cleaned_df.isnull().sum().rename("Missing (clean)")
    summary = pd.concat([raw_missing, clean_missing], axis=1)
    summary.index.name = "Column"
    st.dataframe(summary, use_container_width=True)

    st.caption(
        f"Duplicates removed: **{raw_df.duplicated().sum()}**  |  "
        f"watch_duration_min capped at **300 min** (raw max = "
        f"{raw_df['watch_duration_min'].max():.0f} min)"
    )


# ── Page 2: Warehouse & OLAP ──────────────────────────────────────────────────
elif page == "Warehouse & OLAP":
    st.title("Warehouse & OLAP  (Module I)")
    show_banner()

    fact_df, dims = get_schema(cleaned_df)

    st.subheader("Star Schema — Fact table")
    st.write(f"**Fact_Streams** — {len(fact_df):,} rows")
    st.dataframe(fact_df.head(10), use_container_width=True)

    st.subheader("Dimension tables")
    dcols = st.columns(len(dims))
    for col, (name, ddf) in zip(dcols, dims.items()):
        col.write(f"**{name}** ({len(ddf)} rows)")
        col.dataframe(ddf, use_container_width=True)

    rollup, premium_genre, _ = get_olap(cleaned_df)
    _, _, dice = get_olap(df_filtered)

    st.subheader("Roll-up: avg watch minutes by Tier × Genre")
    st.caption("OLAP operation: Roll-Up — aggregates streaming minutes across genre and tier dimensions.")
    if not rollup.empty:
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(rollup.fillna(0), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(rollup.columns))); ax.set_xticklabels(rollup.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(rollup.index)));   ax.set_yticklabels(rollup.index)
        for i in range(len(rollup.index)):
            for j in range(len(rollup.columns)):
                v = rollup.iloc[i, j]
                ax.text(j, i, f"{v:.0f}" if not np.isnan(v) else "", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Avg watch min"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.subheader("Slice: mean monthly spend by Genre (Premium subscribers only)")
    st.caption("OLAP operation: Slice — fixes subscription_tier = 'Premium', views spend across genres.")
    if len(premium_genre) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        premium_genre.plot(kind="bar", ax=ax2, color="#4C72B0", edgecolor="white")
        ax2.set_xlabel("Genre"); ax2.set_ylabel("Mean monthly spend ($)"); ax2.set_xticklabels(premium_genre.index, rotation=45, ha="right")
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    else:
        st.info("No Premium subscribers present.")

    st.subheader("Dice: total monthly spend by Tier × Country (filtered selection)")
    st.caption("OLAP operation: Dice — filters both tier and country dimensions simultaneously.")
    if len(dice) > 0:
        dice_df = dice.reset_index()
        dice_df.columns = ["Tier", "Country", "Total Spend ($)"]
        st.dataframe(dice_df, use_container_width=True)
    else:
        st.warning("No data for current filter selection.")


# ── Page 3: Outliers ──────────────────────────────────────────────────────────
elif page == "Outliers":
    st.title("Outlier Detection  (Module IV)")
    show_banner()

    out = get_outliers(raw_df)

    kpi_row([
        ("IQR outliers",              str(out["iqr_count"])),
        ("IQR lower fence",           f"{out['iqr_lower']:.1f} min"),
        ("IQR upper fence",           f"{out['iqr_upper']:.1f} min"),
        ("Isolation Forest outliers", str(out["iso_count"])),
    ])

    st.subheader("Box plot of raw watch_duration_min")
    col = raw_df["watch_duration_min"].dropna()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.boxplot(col, vert=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_xlabel("watch_duration_min (minutes)")
    ax.axvline(out["iqr_lower"], color="red",   linestyle="--", label=f"IQR lower ({out['iqr_lower']:.0f})")
    ax.axvline(out["iqr_upper"], color="orange", linestyle="--", label=f"IQR upper ({out['iqr_upper']:.0f})")
    ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("IQR flagged values"):
        st.write(sorted(out["iqr_values"]))
    with st.expander("Isolation Forest flagged values (sample, 20)"):
        st.write(sorted(out["iso_values"])[:20])


# ── Page 4: Segments ──────────────────────────────────────────────────────────
elif page == "Segments":
    st.title("Subscriber Segmentation  (Module IV — Clustering)")
    show_banner()

    seg = get_segments(cleaned_df)

    st.subheader("Silhouette score by k")
    st.caption(f"Best k = **{seg['best_k']}** (highest silhouette score)")
    sil_df = pd.DataFrame(list(seg["silhouette_by_k"].items()), columns=["k", "Silhouette"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(sil_df["k"], sil_df["Silhouette"], marker="o", color="#4C72B0")
    ax.axvline(seg["best_k"], color="red", linestyle="--", label=f"Best k={seg['best_k']}")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score"); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader(f"K-Means clusters (k={seg['best_k']}) — PCA projection")
    coords  = np.array(seg["pca_coords"])
    labels  = np.array(seg["cluster_labels"])
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for c in range(seg["best_k"]):
        mask = labels == c
        ax2.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.6,
                    label=f"Cluster {c} (n={seg['cluster_sizes'][c]})", color=colors[c % len(colors)])
    ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2"); ax2.legend()
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    st.caption(f"DBSCAN (eps=1.5, min_samples=5) noise points: **{seg['dbscan_noise']}**")

    st.subheader("Cluster profile (mean feature values)")
    profile_df = pd.DataFrame(seg["cluster_profile"])
    profile_df.index.name = "Cluster"
    st.dataframe(profile_df, use_container_width=True)


# ── Page 5: Churn model ────────────────────────────────────────────────────────
elif page == "Churn model":
    st.title("Churn Classification  (Module III)")
    show_banner()

    st.info(
        "**Churn definition:** `completion_pct < 50` (subscriber watched less than "
        "half of content on average) is labelled **Churned = 1**."
    )
    st.caption(
        "Accuracy is close to chance because the synthetic features carry little "
        "signal about completion. The point of this page is the method and the "
        "evaluation, not a strong predictor."
    )

    results, fitted_dt, feature_names = get_churn(cleaned_df)

    table_rows = []
    for model_name, r in results.items():
        table_rows.append({
            "Model":            model_name,
            "Holdout acc.":     f"{r['accuracy']:.3f}",
            "Churn prec.":      f"{r['precision']:.3f}",
            "Churn recall":     f"{r['recall']:.3f}",
            "Churn F1":         f"{r['f1']:.3f}",
            "CV acc. (5-fold)": f"{r['cv_accuracy']:.3f}",
        })
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(table_rows).set_index("Model"), use_container_width=True)

    st.subheader("Confusion matrix")
    model_choice = st.selectbox("Choose model", list(results.keys()))
    cm = results[model_choice]["confusion_matrix"]
    fig = make_confusion_heatmap(cm, ["Retained (0)", "Churned (1)"])
    st.pyplot(fig); plt.close(fig)

    with st.expander("Decision Tree text rules (sklearn export_text, depth 3)"):
        from sklearn.tree import export_text
        tree_text = export_text(fitted_dt, feature_names=feature_names, max_depth=3)
        st.code(tree_text, language="")


# ── Page 6: Recommendations ────────────────────────────────────────────────────
elif page == "Recommendations":
    st.title("Genre Recommendations  (Modules V & VI — Association Rules)")
    show_banner()

    st.info(
        "The rules mostly reflect co-viewing patterns planted in the synthetic "
        "generator. They are not real findings."
    )

    if not os.path.exists(_CSV_CA3):
        with st.spinner("Generating viewing sessions dataset..."):
            from generate_viewing_sessions import generate
            df_gen = generate()
            os.makedirs(os.path.dirname(_CSV_CA3), exist_ok=True)
            df_gen.to_csv(_CSV_CA3, index=False)

    c1, c2, c3 = st.columns(3)
    min_sup  = c1.slider("Min support",    0.01, 0.20, 0.05, 0.01)
    min_conf = c2.slider("Min confidence", 0.10, 0.90, 0.40, 0.05)
    min_lift = c3.slider("Min lift",       1.0,  5.0,  1.1,  0.1)

    sessions, baskets, rules = get_rules(min_sup, min_conf, min_lift)

    st.subheader(f"Top 15 rules by lift  ({len(rules)} total)")
    st.dataframe(rules.head(15), use_container_width=True)

    st.subheader("User recommendations")
    all_users = sorted(sessions["user_id"].unique())
    chosen_uid = st.selectbox("Choose a user_id", all_users)

    user_genres = set(
        sessions[sessions["user_id"] == chosen_uid]["genres_watched"]
        .str.split(",")
        .explode()
        .str.strip()
        .unique()
    )
    st.write(f"**Genres watched by user {chosen_uid}:** {', '.join(sorted(user_genres))}")

    recs = recommend(user_genres, rules, top_n=5)
    if recs:
        st.success(f"**Recommended genres:** {', '.join(recs)}")
    else:
        st.info("No recommendations (user has already watched all rule consequents, or no matching rules).")
