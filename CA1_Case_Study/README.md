# CA1 Case Study — Video Streaming Platform BI

> **Submission Evaluation:** Continuous Assessment 1 (CA1) — 10 Marks

---

## Case Study Title

**Diagnosing Subscriber Churn on a Video Streaming Platform: A Case Study**  
*(Note: Title is marked as a placeholder and can be modified by the project team before final submission)*

---

## Submission Details

| Field | Detail / Link |
|---|---|
| **Case Study Title** | Diagnosing Subscriber Churn on a Video Streaming Platform: A Case Study *(Placeholder)* |
| **Presentation (PPT) Link** | `[PLACEHOLDER: Insert Google Slides / OneDrive / Drive Link here]` |
| **Reference Paper Link** | `[PLACEHOLDER: https://doi.org/10.1007/s00607-021-00908-y — Verify before submission]` |
| **Student Name** | `[PLACEHOLDER: Student Name]` |
| **Roll Number** | `[PLACEHOLDER: Roll No. e.g. 40]` |
| **Academic Term / Batch** | DMBI Lab / AY 2026–27 |

---

## Case Study Overview & Scope

This case study investigates subscriber retention, engagement metrics, and churn behavior within an over-the-top (OTT) video streaming platform. It focuses on the business, statistical, and data mining methodologies needed to diagnose why subscribers stop viewing content and cancel recurring subscriptions.

### Core Focus Areas

1. **Business Problem & Context:**
   - Recurring revenue erosion due to user drop-offs.
   - Low completion rates (< 50%) acting as early signals of dissatisfaction.
   - Discrepancies in engagement across subscription tiers (Basic, Standard, Premium).

2. **Data Understanding & Quality Audit:**
   - Audit of synthetic subscriber dataset (`streaming_platform_dataset.csv`, 515 raw records).
   - Identification of 13 duplicate records and missing values across demographic and engagement features.
   - Outlier detection in viewing durations (spurious values up to 9999 minutes flagged and capped).
   - Cleaned cohort of 502 valid subscriber records.

3. **Subscriber Segmentation:**
   - Evaluating subscriber clusters using K-Means with Silhouette analysis (optimal $k=2$).
   - Cluster separation: High-engagement/High-spend power users vs. Low-engagement casual subscribers.
   - DBSCAN density analysis identifying anomaly subscribers.

4. **Predictive Churn Modeling & Diagnostic Evaluation:**
   - Defining churn threshold: completion percentage $< 50\%$.
   - Classification diagnostics utilizing Decision Tree, Gaussian Naïve Bayes, and Random Forest.
   - Analysis of classification metrics (Accuracy, Precision, Recall, F1-score) and confusion matrices.
   - Diagnostic insights on model limitations: weak predictive signal from synthetic features, emphasizing the need for behavioral telemetry data (session frequencies, buffering rates, search failures).

5. **Strategic Recommendations:**
   - Proactive onboarding interventions for low-completion users.
   - Personalized content recommendation engines to boost content discovery.
   - Tier-specific retention incentives.

---

## Repository Contents

```
CA1_Case_Study/
├── README.md                           # Case study documentation & submission metadata
├── data/
│   └── streaming_platform_dataset.csv  # 515 subscriber dataset copy
├── presentation/
│   ├── Diagnosing-Subscriber-Churn.pptx # 16:9 presentation slide deck (8 slides)
│   └── CA1_Case_Study.pptx             # Standard coursework deck alias
└── references/
    └── reference_paper.txt             # Academic literature citation & verification notes
```

---

## Academic Reference

- Lalwani, P., Mishra, M. K., Chadhar, J. S., & Sethi, P. (2022). Customer churn prediction system: a machine learning approach. *Computing*, 104(2), 271–294. https://doi.org/10.1007/s00607-021-00908-y *(Verify before submission)*
