# Experiment 9: FP-Growth Association Rules in RapidMiner (Altair AI Studio) & Python

Course: Data Mining and Business Intelligence (DMBI Lab) — Roll No. 38.

## RapidMiner Process (Design View)
`Retrieve (JioHotstar association transactions) -> Numerical to Binominal -> FP-Growth (min support 0.05) -> Create Association Rules (min confidence 0.30)`

- **Datasets:**
  - `JioHotstar_Preprocessed_Dataset.csv` (33,171 records x 23 columns).
  - `JioHotstar_Association_Transactions.csv` (33,171 transactions x 7 binary attributes).
- **Licence Limit:** Altair AI Studio Free licence processes 10,000 examples.
- **Report Document:** `EXP-9_DMBI_40_FPGrowth_RapidMiner.docx`

## Python Implementation
Run the Python script that implements FP-Growth using `mlxtend` and compares execution performance with Apriori:

```bash
python fpgrowth_association.py
```

### Outputs
- 22 frequent itemsets generated via FP-Tree.
- 25 positive association rules (confidence $\ge 0.30$, lift $\ge 1.0$).
- Execution speedup comparison: FP-Growth vs. Apriori.
