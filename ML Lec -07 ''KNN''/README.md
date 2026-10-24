# Telco Customer Churn: EDA & Baseline Model

A beginner-friendly data science project: explore why telecom customers leave ("churn") and build a first predictive model.

---

## 1. Dataset Overview

- **Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 customers x 21 columns
- **Target variable:** `Churn` (Yes/No). About 26.5% of customers churned, so the classes are **imbalanced**.

| Group | Columns | Type |
|---|---|---|
| Identifier | `customerID` | text (drop before modeling) |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | categorical (`SeniorCitizen` is already 0/1) |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `tenure` numeric, others categorical |
| Billing | `MonthlyCharges`, `TotalCharges` | numeric (`TotalCharges` loads as text!) |
| Phone | `PhoneService`, `MultipleLines` | categorical |
| Internet | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical |
| **Target** | `Churn` | Yes / No |

## 2. Project Goals & Learning Outcomes

By the end, students can:
1. Load, inspect, and audit a real-world tabular dataset.
2. Detect and fix data-type problems and missing values.
3. Perform univariate and bivariate EDA and read the plots correctly.
4. Engineer useful features (tenure bins, service counts, segments).
5. Build and evaluate a baseline classifier with a leak-free pipeline.
6. Translate findings into business recommendations, and state limitations honestly.

## 3. Data Cleaning Steps

1. **Drop `customerID`.** It is unique per row and carries no signal.
2. **Convert `TotalCharges` to numeric** with `pd.to_numeric(..., errors="coerce")`. It loads as `object` because 11 rows contain a blank space.
3. **Understand the 11 missing values.** All have `tenure == 0` (brand-new customers not yet billed). Either set them to 0 or impute with the median, but do it **after** the train/test split (inside a pipeline) to avoid data leakage.
4. **Encode `Churn`** as 1 = Yes, 0 = No.
5. **Check duplicates** (there are none, since `customerID` is unique).
6. **Tidy redundant categories.** `"No internet service"` and `"No phone service"` are just `"No"` in disguise for the add-on columns. Merging them is optional but simplifies the data.
7. **Encode categoricals correctly.**
   - Binary columns: map to 0/1.
   - Multi-class *nominal* columns (`InternetService`, `PaymentMethod`, `Contract`, ...): use **one-hot encoding**, not arbitrary integers. Numbers like `Electronic check = 3, Mailed check = 7` invent an order that does not exist and mislead distance-based models such as KNN.
8. **Scale numeric features** (`tenure`, `MonthlyCharges`, `TotalCharges`) for models sensitive to scale (KNN, logistic regression, SVM).

## 4. Key Analyses & Visualizations to Reproduce

- Churn class balance (bar/pie chart)
- Histograms/KDE for `tenure`, `MonthlyCharges`, `TotalCharges`
- Count plots for every categorical feature
- Churn rate by `Contract`, `InternetService`, `PaymentMethod`
- Boxplots of `tenure` and `MonthlyCharges` split by churn
- Correlation heatmap of numeric features
- Churn rate by tenure bin and by number of add-on services
- Baseline model: confusion matrix, ROC curve, feature importance (coefficients)

The code for every EDA step is in **Section 6** below; the full baseline model is in `telco_churn_eda.ipynb`.

## 5. How to Run

```bash
# 1. Get the data
#    Download from Kaggle and place the CSV next to the notebook (or in ./data/)

# 2. Create an environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch
jupyter notebook telco_churn_eda.ipynb
```

In **Google Colab**: upload the CSV in the Files panel and set `DATA_PATH = "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"` in the first code cell. Then use *Runtime > Run all*.

## 6. EDA Explanation (Lecture Notes, with Code)

Each step below has three parts: **why** we do it, the **code**, and **what to look for**. The code blocks build on each other, so run them in order in one notebook.

### 6.0 Setup (run once)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")          # clean default plot style
pd.set_option("display.max_columns", 50)  # show all columns when printing

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

### 6.1 Data loading and initial inspection

**Why:** before any analysis, confirm the data loaded correctly and understand its size, types, and quality.

```python
# Load the CSV into a DataFrame
df = pd.read_csv(DATA_PATH)

# Shape = (rows, columns)
print("Shape:", df.shape)                      # (7043, 21)

# First rows: do the values look sensible?
df.head()
```

```python
# Column names, dtypes, and non-null counts
df.info()
```

```python
# Summary statistics for numeric columns (count, mean, std, min, quartiles, max)
df.describe()
```

```python
# Quality checks: duplicates and missing values
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customerIDs:", df["customerID"].duplicated().sum())

# Missing values as a percentage per column
(df.isnull().mean() * 100).round(2)
```

```python
# Hidden missing values: blank strings are NOT detected by isnull()
blank_mask = df["TotalCharges"].str.strip() == ""
print("Blank TotalCharges:", blank_mask.sum())          # 11
df.loc[blank_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
```

**What to look for:**
- 7,043 rows x 21 columns, with no duplicates.
- `TotalCharges` is `object` (text), though it should be numeric. `info()` shows no nulls, but the blanks are hiding as `" "`.
- `SeniorCitizen` is 0/1, so its mean (about 0.16) is the share of seniors.
- `tenure` spans 0 to 72 months and `MonthlyCharges` roughly 18 to 119.

### 6.2 Handling missing values and data types

**Why:** wrong types break calculations and plots. Missing values must be handled deliberately.

```python
df_clean = df.copy()   # never modify the raw data; work on a copy

# 1) Convert TotalCharges to numeric. errors="coerce" turns blanks into NaN.
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

# 2) Investigate WHY they are missing before choosing a fix
missing_rows = df_clean[df_clean["TotalCharges"].isna()]
print("Missing TotalCharges:", len(missing_rows))
print("Their tenure values:", missing_rows["tenure"].unique())   # [0] -> brand-new customers

# 3) Fix: they have not been billed yet, so 0 is logically correct.
#    (In the modeling pipeline use SimpleImputer fitted on train data only.)
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)

# 4) Encode the target as 0/1
df_clean["Churn"] = df_clean["Churn"].map({"No": 0, "Yes": 1})

# 5) SeniorCitizen is categorical in meaning: give it readable labels
df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"})

# 6) Merge redundant categories: "No internet/phone service" is really "No"
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
for c in addon_cols + ["MultipleLines"]:
    df_clean[c] = df_clean[c].replace({"No internet service": "No",
                                       "No phone service": "No"})

# 7) Drop the identifier (unique per row, no predictive value)
df_clean = df_clean.drop(columns="customerID")

# Verify
print(df_clean.dtypes.value_counts())
print("Remaining NaNs:", df_clean.isnull().sum().sum())
```

**What to look for:** all 11 missing values come from `tenure == 0`. The cause explains the fix, which is the main lesson here.

### 6.3 Univariate analysis

**Why:** understand each variable on its own (shape, skew, balance) before relating it to churn.

```python
# Split columns by type so we can loop over each group
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in df_clean.columns if c not in num_cols + ["Churn"]]
print(len(num_cols), "numeric,", len(cat_cols), "categorical")
```

**Target variable: how balanced is it?**
```python
print(df_clean["Churn"].value_counts(normalize=True).round(3))

ax = sns.countplot(data=df_clean, x="Churn")
ax.set_xticklabels(["No (0)", "Yes (1)"])
ax.set_title(f"Churn distribution (churn rate = {df_clean['Churn'].mean():.1%})")
plt.show()
```

**Numeric features: histograms with a KDE curve show shape and skew.**
```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, num_cols):
    sns.histplot(df_clean[col], kde=True, bins=30, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# Numeric skewness: values far from 0 mean a lopsided distribution
print(df_clean[num_cols].skew().round(2))
```

**Categorical features: count plots show category sizes.**
```python
n, ncols = len(cat_cols), 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows))
for ax, col in zip(axes.ravel(), cat_cols):
    sns.countplot(data=df_clean, x=col, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
for ax in axes.ravel()[n:]:
    ax.axis("off")                      # hide unused subplot slots
plt.tight_layout()
plt.show()
```

**Quick table of category shares:**
```python
for col in ["Contract", "InternetService", "PaymentMethod"]:
    print(df_clean[col].value_counts(normalize=True).round(3), "\n")
```

**What to look for:**

| Feature | Type | Typical plot | What you see |
|---|---|---|---|
| `Churn` | binary target | count plot | ~73% No / ~27% Yes (imbalanced) |
| `tenure` | numeric | histogram | U-shaped: many new and many long-term customers |
| `MonthlyCharges` | numeric | histogram/KDE | spike at the low end (phone-only plans), then a broad hump |
| `TotalCharges` | numeric | histogram | strongly right-skewed (roughly tenure x monthly) |
| `gender`, `Partner`, `Dependents`, `PaperlessBilling` | binary | count plot | gender balanced; ~half have partners; ~30% dependents |
| `Contract` | ordinal | count plot | month-to-month dominates (~55%) |
| `InternetService` | categorical | count plot | Fiber optic > DSL > No |
| `PaymentMethod` | categorical | bar | four similar groups, electronic check largest |
| Add-ons | categorical | count plot | most customers lack security/support add-ons |

### 6.4 Bivariate analysis (churn vs key features)

**Why:** this is where the insights come from. We compare churners and non-churners.

**Numeric features vs churn (tenure, monthly charges, total charges):**
```python
# Boxplots: compare the distribution for churn=0 vs churn=1
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(data=df_clean, x="Churn", y=col, ax=ax)
    ax.set_xticklabels(["No", "Yes"])
    ax.set_title(f"{col} by churn")
plt.tight_layout()
plt.show()

# Exact numbers behind the boxplots
df_clean.groupby("Churn")[num_cols].agg(["mean", "median"]).round(1)
```

**Churn rate by category: a reusable helper function.**
```python
def churn_rate_plot(col, ax=None, order=None):
    """Bar chart of churn rate (%) per category, with the overall rate as a red line."""
    rate = df_clean.groupby(col)["Churn"].mean().mul(100)
    if order:
        rate = rate.reindex(order)
    ax = ax or plt.gca()
    sns.barplot(x=rate.index, y=rate.values, ax=ax)
    ax.axhline(df_clean["Churn"].mean() * 100, color="red", ls="--", label="overall")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.set_title(f"Churn rate by {col}")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.legend()

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
churn_rate_plot("Contract", axes[0], ["Month-to-month", "One year", "Two year"])
churn_rate_plot("InternetService", axes[1])
churn_rate_plot("PaymentMethod", axes[2])
plt.tight_layout()
plt.show()
```

**Other features (add-ons, demographics, billing):**
```python
other = ["OnlineSecurity", "TechSupport", "PaperlessBilling",
         "Partner", "Dependents", "SeniorCitizen"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), other):
    churn_rate_plot(col, ax)
plt.tight_layout()
plt.show()
```

**Tenure in detail: churn rate for each month of tenure.**
```python
tenure_rate = df_clean.groupby("tenure")["Churn"].mean()
plt.figure(figsize=(10, 4))
sns.lineplot(x=tenure_rate.index, y=tenure_rate.values)
plt.title("Churn rate by tenure (months)")
plt.ylabel("Churn rate")
plt.show()
```

**Correlation heatmap (numeric and 0/1 columns):**
```python
corr_df = df_clean[num_cols + ["Churn"]].copy()
for c in ["Partner", "Dependents", "PaperlessBilling", "PhoneService"]:
    corr_df[c] = df_clean[c].map({"No": 0, "Yes": 1})

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.show()
```

**What to look for:**
- **Tenure:** churners cluster at low tenure. The first months are the riskiest.
- **MonthlyCharges:** churners pay more on average, especially in the 70 to 100 range.
- **TotalCharges:** churners have lower totals, mostly because of short tenure. It is a *proxy* for tenure (correlation with `tenure` is about 0.83).
- **Contract:** the strongest signal. Roughly 43% churn on month-to-month vs about 11% on one-year vs about 3% on two-year.
- **InternetService:** Fiber optic about 42%, DSL about 19%, no internet about 7%.
- **PaymentMethod:** electronic check about 45% vs about 15 to 19% for the others.
- **Add-ons:** customers without `OnlineSecurity` or `TechSupport` churn much more.

(Figures are approximate. Students should confirm them from the output.)

### 6.5 Feature engineering ideas

**Why:** new features can expose patterns that raw columns hide.

```python
fe = df_clean.copy()

# 1) Tenure bins: turn a continuous variable into lifecycle stages
fe["tenure_bin"] = pd.cut(fe["tenure"], bins=[-1, 12, 24, 48, 72],
                          labels=["0-12", "13-24", "25-48", "49-72"])

# 2) Family vs single
fe["has_family"] = ((fe["Partner"] == "Yes") | (fe["Dependents"] == "Yes")).astype(int)

# 3) Number of add-on services subscribed
fe["n_addons"] = (fe[addon_cols] == "Yes").sum(axis=1)

# 4) Average monthly spend (guard against division by zero when tenure == 0)
fe["avg_monthly"] = np.where(fe["tenure"] > 0,
                             fe["TotalCharges"] / fe["tenure"],
                             fe["MonthlyCharges"])

# 5) Price change signal: current price minus historical average
fe["price_gap"] = fe["MonthlyCharges"] - fe["avg_monthly"]

# 6) Auto-pay flag
fe["autopay"] = fe["PaymentMethod"].str.contains("automatic").astype(int)

# 7) Interaction features: combine two risk factors
fe["m2m_fiber"] = ((fe["Contract"] == "Month-to-month") &
                   (fe["InternetService"] == "Fiber optic")).astype(int)
fe["m2m_echeck"] = ((fe["Contract"] == "Month-to-month") &
                    (fe["PaymentMethod"] == "Electronic check")).astype(int)
```

**Check whether the new features are useful: churn rate by segment.**
```python
for col in ["tenure_bin", "has_family", "n_addons", "autopay", "m2m_fiber", "m2m_echeck"]:
    print(fe.groupby(col, observed=True)["Churn"].agg(["mean", "count"]).round(3), "\n")
```

```python
# Cross-segment heatmap: Contract x InternetService
pivot = fe.pivot_table(index="Contract", columns="InternetService",
                       values="Churn", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds")
plt.title("Churn rate: Contract x InternetService")
plt.show()
```

**What to look for:** a feature is useful if the churn rate **differs clearly between its groups** (and each group has enough customers). For example, `m2m_fiber` isolates a very high-risk segment, and churn falls as `n_addons` rises.

### 6.6 Observations, insights, and business interpretation

**Why:** turn charts into decisions. This code produces one summary table students can cite.

```python
# Churn rate and customer count for the key segments, ranked by risk
segments = {
    "Month-to-month contract": fe["Contract"] == "Month-to-month",
    "Two-year contract":       fe["Contract"] == "Two year",
    "Fiber optic":             fe["InternetService"] == "Fiber optic",
    "Electronic check":        fe["PaymentMethod"] == "Electronic check",
    "No TechSupport":          fe["TechSupport"] == "No",
    "Tenure 0-12 months":      fe["tenure_bin"] == "0-12",
    "Tenure 49-72 months":     fe["tenure_bin"] == "49-72",
    "Senior citizen":          fe["SeniorCitizen"] == "Yes",
    "Has family":              fe["has_family"] == 1,
}
summary = pd.DataFrame({
    "customers":  {k: int(m.sum()) for k, m in segments.items()},
    "churn_rate": {k: fe.loc[m, "Churn"].mean() for k, m in segments.items()},
})
summary["vs_overall"] = summary["churn_rate"] / fe["Churn"].mean()   # lift
summary.sort_values("churn_rate", ascending=False).round(3)
```

```python
# How much of all churn does each risky segment capture? (helps prioritize)
total_churners = fe["Churn"].sum()
for name in ["Month-to-month contract", "Tenure 0-12 months", "Fiber optic"]:
    m = segments[name]
    print(f"{name}: {fe.loc[m, 'Churn'].sum() / total_churners:.0%} of all churners")
```

**Business interpretation:**
1. **Early-life churn is the biggest problem.** Invest in onboarding during months 0 to 12.
2. **Contracts retain.** Offer incentives to move month-to-month customers to annual plans.
3. **Fiber customers churn heavily.** Investigate price, reliability, or competitor offers.
4. **Support and security add-ons correlate with loyalty.** Bundle or trial them.
5. **Electronic check users churn most.** Promote auto-pay (for example, a small discount).
6. **Higher bills, higher risk.** Review price sensitivity at the 70+ monthly tier.

### 6.7 Limitations and potential biases

**Why:** a good analyst checks the data's weaknesses. These quick checks make the caveats concrete.

```python
# 1) Class imbalance: accuracy of a model that ALWAYS predicts "No churn"
majority_acc = 1 - fe["Churn"].mean()
print(f"Always-'No' accuracy: {majority_acc:.1%}  <- any model must beat this")
```

```python
# 2) Collinearity: tenure, MonthlyCharges and TotalCharges overlap
print(fe[num_cols].corr().round(2))
# TotalCharges is almost "tenure x MonthlyCharges"
print("corr(TotalCharges, tenure*MonthlyCharges):",
      round(np.corrcoef(fe["TotalCharges"], fe["tenure"] * fe["MonthlyCharges"])[0, 1], 3))
```

```python
# 3) Small segments: churn rates from few customers are unreliable
seg_sizes = fe.groupby(["Contract", "InternetService", "PaymentMethod"]).size()
print("Smallest segments:\n", seg_sizes.sort_values().head(5))
```

```python
# 4) Fairness check: do demographic groups differ in churn? (do NOT use blindly in decisions)
print(fe.groupby("gender")["Churn"].mean().round(3))
print(fe.groupby("SeniorCitizen")["Churn"].mean().round(3))
```

**Key limitations:**
- **Correlation is not causation.** Add-ons may reflect engaged customers rather than cause loyalty.
- **Snapshot data.** One point in time; no churn timing, seasonality, or trends.
- **Synthetic/sample origin.** IBM sample data may not reflect real market behavior.
- **Class imbalance.** Accuracy alone is misleading; use recall, precision, and ROC-AUC.
- **Missing drivers.** No data on competitors, complaints, call-center contacts, network quality, or reasons for leaving.
- **Demographic fields** (`gender`, `SeniorCitizen`) must be handled carefully to avoid unfair or discriminatory decisions.
- **Redundant features** (`tenure`, `TotalCharges`, `MonthlyCharges`) are collinear, so interpret coefficients cautiously.
- A single train/test split gives a noisy estimate; use cross-validation.

## 7. Interpreting Results & Caveats

- Compare any model to the **majority-class baseline (about 73.5% accuracy)**. Beating it matters, not the raw number.
- Prefer **recall, precision, F1, and ROC-AUC** for the churn class. Business context decides the trade-off (missing a churner vs wasting a retention offer).
- A typical logistic regression baseline reaches about 80% accuracy and ROC-AUC around 0.84.
- Notice whether tuning helps or the model is already near the data's ceiling.

## 8. Common Pitfalls (from student notebooks)

| Pitfall | Fix |
|---|---|
| Arbitrary integers for nominal categories (`PaymentMethod` -> 3, 5, 6, 7) | One-hot encode |
| Filling NaNs on the full dataset before splitting | Impute inside a `Pipeline` |
| KNN on unscaled features (`TotalCharges` dwarfs binary columns) | `StandardScaler` |
| Reporting accuracy only | Add confusion matrix, recall, ROC-AUC |
| Random split on imbalanced target | `stratify=y` |

## 9. References

- Kaggle dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- pandas docs: https://pandas.pydata.org/docs/
- seaborn tutorial: https://seaborn.pydata.org/tutorial.html
- scikit-learn Pipelines & ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
- Imbalanced classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
