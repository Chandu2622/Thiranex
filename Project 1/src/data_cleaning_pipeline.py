"""
Data Cleaning & Visualization Project
======================================
Dataset: Titanic passenger dataset (public, via seaborn)
Goal:
  - Load a raw/messy dataset
  - Handle missing values, outliers, and duplicates
  - Explore and visualize key insights
  - Save a cleaned dataset + charts for a visual report

Run: python3 data_cleaning_pipeline.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/outputs_assets"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------
print("Loading raw Titanic dataset...")
df_raw = sns.load_dataset("titanic")

# To realistically demonstrate duplicate-handling skills (the public Titanic
# dataset ships with no duplicate rows), we inject a handful of duplicate
# records -- exactly the kind of thing that turns up in real "raw exports".
np.random.seed(42)
dup_rows = df_raw.sample(8, random_state=42)
df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

print(f"Raw shape: {df_raw.shape}")
raw_summary = {
    "rows": df_raw.shape[0],
    "columns": df_raw.shape[1],
    "duplicate_rows": df_raw.duplicated().sum(),
    "missing_cells": df_raw.isna().sum().sum(),
}
print(raw_summary)

# ---------------------------------------------------------------------------
# 2. MISSING VALUES — audit
# ---------------------------------------------------------------------------
missing = df_raw.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_raw) * 100).round(1)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0]
print("\nMissing value report:\n", missing_report)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=missing_report.index, y=missing_report["missing_pct"], ax=ax, color="#c0392b")
ax.set_ylabel("% missing")
ax.set_title("Missing Values by Column (Raw Data)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/01_missing_values.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. CLEANING
# ---------------------------------------------------------------------------
df = df_raw.copy()

# 3a. Duplicates
n_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {n_dupes} duplicate rows.")

# 3b. Drop columns that are mostly missing / redundant for analysis
# 'deck' is >75% missing; 'embark_town' duplicates 'embarked'
cols_to_drop = [c for c in ["deck", "embark_town", "alive"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# 3c. Impute missing values
#   - age: median by (sex, pclass) group -> more accurate than a flat median
df["age"] = df.groupby(["sex", "pclass"])["age"].transform(lambda s: s.fillna(s.median()))
#   - embarked: fill with the mode (most common port)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
#   - fare: fill any missing with median fare for that class
if df["fare"].isna().any():
    df["fare"] = df.groupby("pclass")["fare"].transform(lambda s: s.fillna(s.median()))

assert df.isna().sum().sum() == 0, "There are still missing values!"
print("Missing values after cleaning: 0")

# 3d. Outliers — fare has a long right tail (luxury tickets).
q1, q3 = df["fare"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
n_outliers = (df["fare"] > upper_bound).sum()
print(f"\nFare outliers (> {upper_bound:.2f}): {n_outliers} rows")

# Rather than deleting these passengers (they're real, valid data — just
# extreme), we cap ("winsorize") fare at the upper bound to reduce skew
# while preserving every record for the analysis.
df["fare_capped"] = np.where(df["fare"] > upper_bound, upper_bound, df["fare"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=df["fare"], ax=axes[0], color="#e67e22")
axes[0].set_title("Fare — Before Capping (Outliers Visible)")
sns.boxplot(x=df["fare_capped"], ax=axes[1], color="#27ae60")
axes[1].set_title("Fare — After Capping Outliers")
plt.tight_layout()
plt.savefig(f"{OUT}/02_outliers_fare.png", dpi=150)
plt.close()

# Final cleaned dataset saved to CSV
df.to_csv(f"{OUT}/titanic_cleaned.csv", index=False)
print(f"\nCleaned shape: {df.shape}")

# ---------------------------------------------------------------------------
# 4. VISUAL INSIGHTS
# ---------------------------------------------------------------------------

# 4a. Survival rate by passenger class and sex
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=ax, palette="Set2")
ax.set_title("Survival Rate by Class and Sex")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/03_survival_by_class_sex.png", dpi=150)
plt.close()

# 4b. Age distribution before vs after imputation, split by survival
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="survived", multiple="stack", bins=30, palette="Set1", ax=ax)
ax.set_title("Age Distribution by Survival (Post-Cleaning)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_age_distribution.png", dpi=150)
plt.close()

# 4c. Correlation heatmap of numeric features
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare_capped"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{OUT}/05_correlation_heatmap.png", dpi=150)
plt.close()

# 4d. Embarkation port vs survival
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x="embarked", hue="survived", palette="Set2", ax=ax)
ax.set_title("Survival Count by Port of Embarkation")
ax.set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")
plt.tight_layout()
plt.savefig(f"{OUT}/06_embarked_survival.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. KEY FINDINGS (printed — used later in report)
# ---------------------------------------------------------------------------
overall_survival = df["survived"].mean() * 100
female_survival = df.loc[df.sex == "female", "survived"].mean() * 100
male_survival = df.loc[df.sex == "male", "survived"].mean() * 100
class1_survival = df.loc[df.pclass == 1, "survived"].mean() * 100
class3_survival = df.loc[df.pclass == 3, "survived"].mean() * 100

findings = f"""
KEY FINDINGS
------------
Overall survival rate: {overall_survival:.1f}%
Female survival rate:  {female_survival:.1f}%
Male survival rate:    {male_survival:.1f}%
1st class survival:    {class1_survival:.1f}%
3rd class survival:    {class3_survival:.1f}%
Fare-survival correlation: {corr.loc['survived','fare_capped']:.2f}
Age-survival correlation:  {corr.loc['survived','age']:.2f}
"""
print(findings)

with open(f"{OUT}/summary_stats.txt", "w") as f:
    f.write(f"RAW DATA SUMMARY\n{raw_summary}\n\n")
    f.write(f"MISSING VALUES (raw)\n{missing_report.to_string()}\n\n")
    f.write(f"Duplicates removed: {n_dupes}\n")
    f.write(f"Fare outliers capped: {n_outliers} (threshold {upper_bound:.2f})\n")
    f.write(findings)

print("\nAll charts + cleaned data + stats saved to", OUT)
