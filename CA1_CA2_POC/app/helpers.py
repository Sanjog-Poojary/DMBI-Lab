"""helpers.py — shared utilities for the Streamlit dashboard."""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BANNER = (
    ":warning: **Synthetic dataset for coursework.** "
    "Results illustrate the method and are not real viewer behaviour."
)


def show_banner():
    st.info(BANNER)


def add_src_to_path():
    """Add CA1_CA2_POC/src and CA3/src to sys.path so imports work."""
    r_root = repo_root()
    for sub in ("CA1_CA2_POC/src", "CA3/src"):
        p = os.path.join(r_root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return r_root


def repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def fmt_number(v, decimals=2):
    """Return a nicely formatted number string."""
    if isinstance(v, float):
        return f"{v:,.{decimals}f}"
    return f"{v:,}"


def kpi_row(labels_values: list[tuple[str, str]]):
    """Render a row of KPI metric cards."""
    cols = st.columns(len(labels_values))
    for col, (label, value) in zip(cols, labels_values):
        col.metric(label, value)


def make_confusion_heatmap(cm, labels):
    """Return a matplotlib Figure of a confusion matrix heatmap (no seaborn)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm[0][0] / 1.5 else "black",
                    fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def apply_sidebar_filters(df: pd.DataFrame):
    """Render tier + country sidebar filters and return filtered df."""
    tiers = sorted(df["subscription_tier"].dropna().unique())
    countries = sorted(df["country"].dropna().unique())

    sel_tiers = st.sidebar.multiselect(
        "Subscription tier", tiers, default=tiers, key="tier_filter"
    )
    sel_countries = st.sidebar.multiselect(
        "Country", countries, default=countries, key="country_filter"
    )

    mask = df["subscription_tier"].isin(sel_tiers) & df["country"].isin(sel_countries)
    return df[mask].copy()

