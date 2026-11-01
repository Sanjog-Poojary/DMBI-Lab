# CA3 Content Beyond Syllabus — Matrix Factorization Recommender

> **Submission Evaluation:** Continuous Assessment 3 (CA3) — 10 Marks

---

## Presentation Title

**Beyond Association Rules: Matrix Factorization for Content Recommendation on Streaming Platforms**

---

## Submission Details

| Field | Detail / Link |
|---|---|
| **Presentation Title** | Beyond Association Rules: Matrix Factorization for Content Recommendation on Streaming Platforms |
| **Presentation (PPT) Link** | `[PLACEHOLDER: Insert Google Slides / PPT Link here]` |
| **Reference Paper Link** | https://doi.org/10.1109/MC.2009.263 *(Koren et al., 2009 — verify before submission)* |
| **Student Name** | `[PLACEHOLDER: Student Name]` |
| **Roll Number** | `[PLACEHOLDER: Roll No. e.g. 40]` |
| **Academic Term / Batch** | DMBI Lab / AY 2026–27 |

---

## Syllabus vs. Beyond Syllabus Context

| Aspect | Syllabus Coverage (Modules V & VI) | Beyond Syllabus (CA3 Contribution) |
|---|---|---|
| **Paradigm** | Association Rule Mining (Apriori Market Basket Analysis) | Collaborative Filtering via Latent Factor Modeling |
| **Mathematical Basis** | Itemset support, rule confidence, and lift thresholds | Matrix Factorization ($R \approx U \cdot V^T$) via Alternating Least Squares (ALS) / SVD |
| **Representation** | Symbolic, discrete antecedent $\rightarrow$ consequent co-occurrence rules | Continuous low-rank dense embedding vectors ($k=8$ latent dimensions) |
| **Cold / Sparse Generalization** | Fails when no predefined rule antecedent matches user viewing history | Generalizes across user-item interaction space through learned latent affinities |
| **Role in Experiment** | **Baseline recommender** imported from `CA2_Mini_Project/src/association_rules.py` | **Target advanced model** implemented in `CA3/src/matrix_factorization.py` |

---

## Folder Layout

```
CA3/
├── README.md                            # Documentation, syllabus justification, submission fields
├── data/
│   └── viewing_sessions.csv             # 2,755 viewing sessions across 500 subscribers
├── src/
│   ├── generate_viewing_sessions.py     # Session data generator
│   ├── association_rules.py             # Syllabus baseline: Apriori rule generation & recommender
│   └── matrix_factorization.py          # Beyond syllabus: NumPy ALS Low-Rank Matrix Factorization
├── outputs/
│   ├── association_rules.csv            # Mined association rules
│   └── cf_results.json                  # Hit Rate @ 3 benchmark evaluation results
├── presentation/
│   └── CA3_Matrix_Factorization.pptx    # 16:9 presentation slide deck (8-10 slides)
└── references/
    └── reference_paper.txt              # Koren et al. (2009) IEEE Computer citation
```

---

## Run Instructions

### 1. Run the Matrix Factorization Recommender & Benchmark

From the repository root:

```bash
python CA3/src/matrix_factorization.py
```

The script:
1. Loads user viewing histories from `CA3/data/viewing_sessions.csv`.
2. Assembles the binary user-genre interaction matrix (500 users $\times$ 8 genres).
3. Splits observed positive interactions into train (80%) and test (20%) sets (`random_state=42`).
4. Trains a low-rank matrix factorization model using NumPy Alternating Least Squares (rank = 8, L2 regularization = 0.1).
5. Evaluates Hit Rate @ 3 against the Popularity baseline and the Apriori Association Rules baseline.
6. Saves comprehensive evaluation metrics to `CA3/outputs/cf_results.json`.

### 2. (Optional) Run the Association Rules Baseline

```bash
python CA3/src/association_rules.py
```

---

## Reference Paper

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37. https://doi.org/10.1109/MC.2009.263 *(Verify before submission)*
