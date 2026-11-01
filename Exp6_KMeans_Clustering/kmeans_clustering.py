"""Experiment 6: K-Means clustering of the JioHotstar dataset (Python).

Notebook cells assembled into one script. Put the dataset CSV in this folder as
JioHotstar_Preprocessed_Dataset.csv and run:  python kmeans_clustering.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JioHotstar_Preprocessed_Dataset.csv")

# --- Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)

features = [
    'episode_duration_min',
    'pacing_score',
    'hook_strength',
    'visual_intensity',
    'avg_watch_percentage',
    'pause_count',
    'rewind_count',
    'skip_intro',
    'cognitive_load',
    'night_watch_safe'
]

X = df[features]
print("Selected Features:")
print(features)
print("\nShape of selected data:", X.shape)
print("\nFirst 5 rows of selected features:")
print(X.head())

# --- Standardize
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("Standardization completed successfully!")
print("\nStandardized data shape:", X_scaled.shape)
print("\nFirst 5 rows of standardized data:")
print(X_scaled.head())

# --- Elbow method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_values = range(2, 11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_values, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for JioHotstar Dataset")
plt.xticks(K_values)
plt.grid(True)
plt.show()

# --- Apply K-Means (K = 4)
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df['Cluster'] = cluster_labels

print("K-Means clustering completed successfully!")
print("\nNumber of records in each cluster:")
print(df['Cluster'].value_counts().sort_index())

# --- Evaluation
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score
silhouette = silhouette_score(X_scaled, cluster_labels)

# Inertia
inertia_value = kmeans.inertia_

# Davies-Bouldin Score
davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

print("Clustering Evaluation Metrics")
print("-------------------------------")
print("Number of Clusters (K):", 4)
print("Silhouette Score:", silhouette)
print("Inertia:", inertia_value)
print("Davies-Bouldin Score:", davies_bouldin)

# --- Visualize
plt.figure(figsize=(10, 6))

plt.scatter(
    df['avg_watch_percentage'],
    df['cognitive_load'],
    c=df['Cluster'],
    cmap='viridis',
    alpha=0.6
)

plt.xlabel("Average Watch Percentage")
plt.ylabel("Cognitive Load")
plt.title("K-Means Clustering of JioHotstar Data")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# --- Cluster characteristics
cluster_summary = df.groupby('Cluster')[features].mean()

print("Mean Characteristics of Each Cluster:")
print(cluster_summary.round(2))
