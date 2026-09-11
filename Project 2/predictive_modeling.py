"""
Predictive Modeling Using Machine Learning
============================================
Task: Predict Titanic passenger survival (binary classification)
Dataset: titanic_cleaned.csv (output of Project 1 - Data Cleaning & Visualization)

Models compared:
  - Logistic Regression   (linear model -- the classification counterpart of
                            Linear Regression, since survival is a 0/1 label)
  - Decision Tree Classifier
  - Random Forest Classifier

Evaluation:
  - Train/test split, accuracy, precision, recall, F1
  - Confusion matrices for all 3 models
  - ROC curves + AUC for all 3 models
  - Feature importance (tree-based models)

Run: python3 predictive_modeling.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/outputs_assets2"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD CLEANED DATA (from Project 1)
# ---------------------------------------------------------------------------
DATA_PATH = "/home/claude/outputs_assets/titanic_cleaned.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded cleaned dataset: {df.shape}")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# Select predictive features and one-hot encode categoricals
features = ["pclass", "sex", "age", "sibsp", "parch", "fare_capped", "embarked"]
target = "survived"

model_df = df[features + [target]].copy()
model_df = pd.get_dummies(model_df, columns=["sex", "embarked"], drop_first=True)

X = model_df.drop(columns=[target])
y = model_df[target]

print(f"\nFeatures used ({X.shape[1]}): {list(X.columns)}")

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Scale features for the linear model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# ---------------------------------------------------------------------------
# 5. MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------
comparison = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"], "Precision": r["precision"],
        "Recall": r["recall"], "F1 Score": r["f1"],
    } for name, r in results.items()
}).T.round(3)
print("\nMODEL COMPARISON\n", comparison)
comparison.to_csv(f"{OUT}/model_comparison.csv")

# Bar chart of metrics
fig, ax = plt.subplots(figsize=(9, 5))
comparison.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/01_model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. CONFUSION MATRICES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"], cbar=False)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/02_confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. ROC CURVES
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
    results[name]["auc"] = roc_auc

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/03_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="crest")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. SAVE SUMMARY
# ---------------------------------------------------------------------------
best_model = comparison["F1 Score"].idxmax()
summary = f"""
PREDICTIVE MODELING SUMMARY
============================
Dataset: {DATA_PATH}
Train/Test split: {X_train.shape[0]} / {X_test.shape[0]} (75/25, stratified)
Features: {list(X.columns)}

MODEL COMPARISON
{comparison.to_string()}

ROC AUC scores:
{chr(10).join(f'  {name}: {r["auc"]:.3f}' for name, r in results.items())}

Best model by F1 score: {best_model}

Top 3 predictive features (Random Forest):
{importances.head(3).to_string()}
"""
print(summary)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(summary)

print(f"\nAll charts and results saved to {OUT}")
