import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                              recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

# ---- Load & clean (same as Experiment 2) ----
data_path = 'streaming_platform_dataset.csv' if os.path.exists('streaming_platform_dataset.csv') else 'dataset/streaming_platform_dataset.csv'
df = pd.read_csv(data_path)
df = df.drop_duplicates()

num_cols = ['age', 'watch_duration_min', 'completion_pct']
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['subscription_tier'] = df['subscription_tier'].astype(str).str.strip().str.title()
df['watch_duration_min'] = df['watch_duration_min'].clip(lower=0, upper=300)

# ---- Target: Churn_Status (Churned if completion below 50%) ----
df['Churn_Status'] = np.where(df['completion_pct'] < 50, 'Churned', 'Retained')
print("Target distribution:")
print(df['Churn_Status'].value_counts())

# ---- Features ----
feature_cols = ['age', 'subscription_tier', 'device_type', 'watch_duration_min',
                'genre', 'country', 'monthly_spend']
X = df[feature_cols].copy()
y = df['Churn_Status'].copy()
for col in ['subscription_tier', 'device_type', 'genre', 'country']:
    X[col] = LabelEncoder().fit_transform(X[col])

label_encoder_y = LabelEncoder()
y_enc = label_encoder_y.fit_transform(y)
classes = list(label_encoder_y.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"\nTraining set: {X_train.shape[0]} rows | Testing set: {X_test.shape[0]} rows")

results = {"classes": classes, "train_size": int(X_train.shape[0]), "test_size": int(X_test.shape[0])}


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    results[name] = {
        "accuracy": round(acc*100, 2),
        "confusion_matrix": cm.tolist(),
        "precision": [round(x*100, 2) for x in prec],
        "recall": [round(x*100, 2) for x in rec],
        "f1": [round(x*100, 2) for x in f1],
    }
    return cm


def save_cm(cm, title, fname, cmap):
    try:
        import seaborn as sns
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes,
                    yticklabels=classes, cbar=False)
    except ImportError:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap=cmap)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ---- Decision Tree ----
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42, min_samples_leaf=10)
dt.fit(X_train, y_train)
cm_dt = evaluate("DecisionTree", y_test, dt.predict(X_test))
save_cm(cm_dt, 'Decision Tree — Confusion Matrix', 'cm_dt.png', 'Blues')

importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:\n", importances.round(4))

plt.figure(figsize=(22, 10))
plot_tree(dt, feature_names=feature_cols, class_names=classes, filled=True,
          rounded=True, fontsize=9, max_depth=3)
plt.title("Decision Tree — Churn_Status (top 3 levels shown)", fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Naive Bayes ----
nb = GaussianNB()
nb.fit(X_train, y_train)
cm_nb = evaluate("NaiveBayes", y_test, nb.predict(X_test))
save_cm(cm_nb, 'Naive Bayes — Confusion Matrix', 'cm_nb.png', 'Blues')

with open('classification_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved decision_tree.png, cm_dt.png, cm_nb.png, classification_summary.json")
