# DMBI Lab — Video Streaming Platform (Roll No. 38)

Repository for Data Mining and Business Intelligence (DMBI) coursework, laboratory practicals, and Continuous Assessment (CA) submissions.

**Domain / Dataset:** Video streaming platform subscriber analytics, viewing sessions, churn behavior, and recommendation systems.

---

## Continuous Assessment (CA) Submissions

| Component | Marks | Title / Topic | Folder |
|---|---|---|---|
| **CA1 Case Study** | 10 | Diagnosing Subscriber Churn on a Video Streaming Platform: A Case Study | [`CA1_Case_Study/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/CA1_Case_Study) |
| **CA2 Mini Project** | 10 | Video Streaming Platform BI: End-to-End Subscriber Analytics, Star Schema OLAP, Churn Diagnostics, and Recommendation Engine | [`CA2_Mini_Project/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/CA2_Mini_Project) |
| **CA3 Content Beyond Syllabus** | 10 | Beyond Association Rules: Matrix Factorization for Content Recommendation on Streaming Platforms | [`CA3/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/CA3) |

---

## Laboratory Experiments (Syllabus Practical Sessions)

| Folder | Experiment | Description & Artifacts |
|---|---|---|
| [`Exp1_Star_Snowflake_Schema/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp1_Star_Snowflake_Schema) | Experiment 1: Schema Design | Dimensional modeling for streaming BI: Star schema vs Snowflake schema design, fact/dimension structures, ER diagrams |
| [`Exp2_Data_Preprocessing/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp2_Data_Preprocessing) | Experiment 2: Data Preprocessing | Handling missing data, deduplication (13 duplicates removed), outlier capping, and data transformation pipeline |
| [`Exp3_EDA_Visualization/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp3_EDA_Visualization) | Experiment 3: EDA & Visualization | Statistical profiling, distribution plots, genre watch-time histograms, and spend correlation matrices |
| [`Exp4_Classification/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp4_Classification) | Experiment 4: Classification Models | Decision Tree (CART) and Gaussian Naïve Bayes classification for subscriber churn prediction |
| [`Exp5_Custom_Classification/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp5_Custom_Classification) | Experiment 5: Custom Classifier | Inbuilt vs from-scratch custom classification functions, evaluation metrics, and comparative benchmarking |
| [`Exp6_KMeans_Clustering/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp6_KMeans_Clustering) | Experiment 6: K-Means Clustering | Customer segmentation and cluster analysis using K-Means algorithm in Python |
| [`Exp7_KMeans_RapidMiner/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp7_KMeans_RapidMiner) | Experiment 7: K-Means RapidMiner | Graphical workflow and cluster validation using RapidMiner Studio |
| [`Exp8_Apriori_Association/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp8_Apriori_Association) | Experiment 8: Apriori Association | Market basket analysis and frequent itemset generation using Apriori algorithm in Python |
| [`Exp9_FP_Growth_RapidMiner/`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/Exp9_FP_Growth_RapidMiner) | Experiment 9: FP-Growth RapidMiner | Frequent pattern mining using FP-Growth and association rule evaluation in RapidMiner |

---

## Quick Start & Running Projects

### Launch the CA2 Streamlit BI Dashboard
- **Windows:** Double-click [`run_app.bat`](file:///c:/Users/Sanjog/Downloads/DMBI-Lab/run_app.bat) or run `run_app.bat` in the terminal.
- **Command Line:**
  ```bash
  python -m streamlit run CA2_Mini_Project/app/app.py
  ```

### Run the CA2 Analysis Pipeline
```bash
python CA2_Mini_Project/src/poc_pipeline.py
```

### Run CA3 Matrix Factorization Recommendation Demo
```bash
python CA3/src/matrix_factorization.py
```

### Run Individual Lab Experiments
Each experiment script can be executed from its respective directory, e.g.:
```bash
# Exp 2 Preprocessing
cd Exp2_Data_Preprocessing && python preprocessing.py

# Exp 6 K-Means Clustering (Python)
cd Exp6_KMeans_Clustering && python kmeans_clustering.py

# Exp 7 K-Means RapidMiner Comparison (Python)
cd Exp7_KMeans_RapidMiner && python kmeans_rapidminer_comparison.py

# Exp 8 Apriori Association Rule Mining (Python)
cd Exp8_Apriori_Association && python apriori_association.py

# Exp 9 FP-Growth Association Rule Mining (Python)
cd Exp9_FP_Growth_RapidMiner && python fpgrowth_association.py
```
