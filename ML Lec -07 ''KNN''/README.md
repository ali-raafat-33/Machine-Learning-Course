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

## 6. EDA Explanation (Lecture Notes)

### 6.1 Data loading and initial inspection
Run `df.shape`, `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()`, `df.duplicated().sum()`.
**What to notice:**
- 7,043 rows x 21 columns.
- `TotalCharges` is `object`, though it should be numeric. `info()` shows no nulls, but the problem is hidden blanks. This is a great lesson that "no missing values" in `info()` does not mean the data is clean.
- `SeniorCitizen` is 0/1 numeric but is really categorical, so its mean (about 0.16) is the share of seniors.
- `describe()`: `tenure` spans 0 to 72 months, `MonthlyCharges` roughly 18 to 119.

### 6.2 Univariate analysis
| Feature | Type | Typical plot | What you see |
|---|---|---|---|
| `Churn` | binary target | count plot | ~73% No / ~27% Yes (imbalanced) |
| `tenure` | numeric | histogram | U-shaped: many brand-new customers and many long-term loyal ones |
| `MonthlyCharges` | numeric | histogram/KDE | a big spike at the low end (phone-only plans), then a broad hump |
| `TotalCharges` | numeric | histogram | strongly right-skewed (it is roughly tenure x monthly) |
| `gender`, `Partner`, `Dependents`, `PaperlessBilling` | binary | count plot | gender is balanced; about half have partners; about 30% have dependents |
| `Contract` | categorical (ordinal) | count plot | month-to-month dominates (~55%) |
| `InternetService` | categorical | count plot | Fiber optic > DSL > No |
| `PaymentMethod` | categorical | horizontal bar | four roughly similar groups, electronic check largest |
| Add-ons (`OnlineSecurity`, `TechSupport`, ...) | categorical | count plot | most customers lack security/support add-ons |

### 6.3 Bivariate analysis (churn vs key features)
- **Tenure:** churners cluster at low tenure (boxplot median is far lower). The first months are the riskiest.
- **MonthlyCharges:** churners pay more on average (higher median), especially in the 70 to 100 range.
- **TotalCharges:** churners have lower totals, but this is mostly because of short tenure. It is a *proxy* for tenure, so watch for redundancy.
- **Contract:** the strongest signal. Roughly 43% churn on month-to-month vs about 11% on one-year vs about 3% on two-year.
- **InternetService:** Fiber optic customers churn about 42%, DSL about 19%, no internet about 7%.
- **PaymentMethod:** electronic check shows about 45% churn vs about 15 to 19% for the other methods.
- **Add-ons:** customers without `OnlineSecurity` or `TechSupport` churn much more.

(Figures are approximate. Have students compute them with `groupby` and compare.)

### 6.4 Handling missing values and data types
- Convert `TotalCharges` to numeric, then investigate the NaNs (`df[df.TotalCharges.isna()]`). All have `tenure == 0`.
- Choose imputation deliberately: `0` is logically sensible (nothing billed yet), median is a safe generic choice.
- Cast `SeniorCitizen` and the target to meaningful types or labels for plotting.
- **Leakage warning:** compute imputation statistics (mean/median) on the training set only.

### 6.5 Feature engineering ideas
- **Tenure bins:** `0-12`, `13-24`, `25-48`, `49-72` months (or quartiles).
- **Churn rate by segment:** `df.groupby("Contract")["Churn"].mean()`, and cross-segments such as Contract x InternetService.
- **Family vs single:** `has_family = (Partner == "Yes") | (Dependents == "Yes")`.
- **Number of add-on services:** sum of "Yes" across the six service columns.
- **Average monthly spend:** `TotalCharges / tenure` (guard against `tenure == 0`) and its gap from `MonthlyCharges` (a pricing-change signal).
- **Interaction features:** `MonthlyCharges x is_month_to_month`, `Fiber x no TechSupport`, `electronic_check x month_to_month`.
- **Auto-pay flag:** `PaymentMethod` contains "automatic".

### 6.6 Observations and business interpretation
1. **Early-life churn is the biggest problem.** Invest in onboarding in months 0 to 12.
2. **Contracts retain.** Offer incentives to migrate month-to-month customers to annual plans.
3. **Fiber customers churn heavily.** Investigate price, reliability, or competitor offers for fiber.
4. **Support and security add-ons correlate with loyalty.** Bundle or trial them.
5. **Electronic check users churn most.** Promote auto-pay (e.g., small discount).
6. **Higher bills, higher risk.** Review price sensitivity at the 70+ monthly tier.

### 6.7 Limitations and potential biases
- **Correlation is not causation.** Add-ons may reflect engaged customers rather than cause loyalty.
- **Snapshot data.** One point in time; no churn timing, seasonality, or trends.
- **Synthetic/sample origin.** IBM sample data may not reflect real market behavior.
- **Class imbalance.** Accuracy alone is misleading; a model predicting "No" always scores about 73%.
- **Missing drivers.** No data on competitors, complaints, call-center contact, network quality, or reasons for leaving.
- **Demographic fields** (`gender`, `SeniorCitizen`) must be handled carefully to avoid unfair or discriminatory decisions.
- **Redundant features** (`tenure`, `TotalCharges`, `MonthlyCharges`) are collinear, so interpret coefficients cautiously.
- Single train/test split gives a noisy estimate; use cross-validation.

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
