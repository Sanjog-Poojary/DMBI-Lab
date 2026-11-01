import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

try:
    import seaborn as sns
except ImportError:
    sns = None

np.random.seed(42)

# ============ 1) LOAD & PREPROCESS (as in Experiment 2) ============
data_path = 'streaming_platform_dataset.csv' if os.path.exists('streaming_platform_dataset.csv') else 'dataset/streaming_platform_dataset.csv'
df = pd.read_csv(data_path)
df = df.drop_duplicates()
for col in ['age', 'watch_duration_min', 'completion_pct']:
    df[col] = df[col].fillna(df[col].mean())
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['subscription_tier'] = df['subscription_tier'].astype(str).str.strip().str.title()
df['watch_duration_min'] = df['watch_duration_min'].clip(lower=0, upper=300)

df['Churn_Status'] = np.where(df['completion_pct'] < 50, 'Churned', 'Retained')

feature_cols = ['age', 'subscription_tier', 'device_type', 'watch_duration_min',
                'genre', 'country', 'monthly_spend']
X = df[feature_cols].copy()
y = df['Churn_Status'].copy()
for col in ['subscription_tier', 'device_type', 'genre', 'country']:
    X[col] = LabelEncoder().fit_transform(X[col])

label_encoder_y = LabelEncoder()
y_enc = label_encoder_y.fit_transform(y)
classes = list(label_encoder_y.classes_)

scale_cols = ['age', 'watch_duration_min', 'monthly_spend']
X[scale_cols] = StandardScaler().fit_transform(X[scale_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

results = {"classes": classes, "train_size": int(X_train.shape[0]), "test_size": int(X_test.shape[0])}


def eval_and_store(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    results[name] = {"accuracy": round(acc*100, 2), "confusion_matrix": cm.tolist()}
    return cm


def save_cm(cm, title, fname, cmap):
    plt.figure(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes,
                    yticklabels=classes, cbar=False)
    else:
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


# ============ 2) INBUILT DECISION TREE (Gini) ============
dt_inbuilt = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, random_state=42)
dt_inbuilt.fit(X_train, y_train)
cm = eval_and_store("DT_Inbuilt", y_test, dt_inbuilt.predict(X_test))
save_cm(cm, 'Decision Tree (Inbuilt, Gini) — Confusion Matrix', 'cm_dt_inbuilt.png', 'Blues')

plt.figure(figsize=(22, 10))
plot_tree(dt_inbuilt, feature_names=feature_cols, class_names=classes,
          filled=True, rounded=True, fontsize=9, max_depth=3)
plt.title("Decision Tree (Inbuilt, Gini, max_depth=5) — top 3 levels shown", fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree_inbuilt.png', dpi=150, bbox_inches='tight')
plt.close()


# ============ 3) CUSTOM DECISION TREE FUNCTION (Entropy) ============
def custom_decision_tree(X_train, y_train, X_test, criterion="entropy", max_depth=5):
    """Student-defined wrapper around DecisionTreeClassifier using entropy."""
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


y_pred_custom, dt_custom_model = custom_decision_tree(X_train, y_train, X_test)
cm = eval_and_store("DT_Custom", y_test, y_pred_custom)
save_cm(cm, 'Custom Decision Tree (Entropy) — Confusion Matrix', 'cm_dt_custom.png', 'Greens')

print("\nText-based decision tree (entropy, depth 4):")
print(export_text(dt_custom_model, feature_names=feature_cols, max_depth=4))


# ============ 4) INBUILT NAIVE BAYES ============
nb_inbuilt = GaussianNB()
nb_inbuilt.fit(X_train, y_train)
cm = eval_and_store("NB_Inbuilt", y_test, nb_inbuilt.predict(X_test))
save_cm(cm, 'Naive Bayes (Inbuilt) — Confusion Matrix', 'cm_nb_inbuilt.png', 'Blues')


# ============ 5) CUSTOM NAIVE BAYES FUNCTION ============
def custom_naive_bayes(X_train, y_train, X_test):
    """Student-defined wrapper around GaussianNB."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model.predict(X_test), model


y_pred_nb_custom, nb_custom_model = custom_naive_bayes(X_train, y_train, X_test)
cm = eval_and_store("NB_Custom", y_test, y_pred_nb_custom)
save_cm(cm, 'Custom Naive Bayes Function — Confusion Matrix', 'cm_nb_custom.png', 'Greens')

with open('exp5_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nAll models trained. Results saved to exp5_results.json")
