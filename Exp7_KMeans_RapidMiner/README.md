# Experiment 7: K-Means Clustering in RapidMiner (Altair AI Studio) & Python Comparison

Course: Data Mining and Business Intelligence (DMBI Lab) — Roll No. 38.

## RapidMiner Process (Design View)
`Retrieve -> Select Attributes -> Replace Missing Values -> Normalize (Z-transformation) -> Clustering (K-Means, k = 4)`

- **Dataset:** `JioHotstar_Preprocessed_Dataset.csv` (33,171 records x 23 columns). Included in this folder.
- **Licence Limit:** Altair AI Studio Free licence processes 10,000 examples.
- **Report Document:** `EXP-7_DMBI_40_KMeans_RapidMiner.docx`

## Python Implementation
Run the Python script that implements the exact RapidMiner workflow and generates the comparative evaluation table:

```bash
python kmeans_rapidminer_comparison.py
```

### Outputs
- Full dataset (33,171 records) and 10,000 slice K-Means metrics.
- Comparative table between Python (scikit-learn) and RapidMiner Studio.
