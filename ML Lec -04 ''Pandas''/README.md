# Complete Pandas Tutorial
A comprehensive tutorial on the Python Pandas library, updated to be consistent with best practices and features available in 2024.

<img src='./images/thumbnail.jpg' width=50%>

The tutorial can be watched [here](https://youtu.be/2uvysYbKdjM?si=8UnGt0bwLwo-eEQL)

The code that is walked through in the tutorial is in [tutorial.ipynb](./tutorial.ipynb)
# 🐼 Pandas for Students: A Practical Guide

A concise, hands-on introduction to **pandas**, Python's most popular library for working with tabular data. This guide takes you from "What is a DataFrame?" to cleaning, transforming, and summarizing real datasets.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [What is pandas?](#1-what-is-pandas)
3. [Quick Start](#2-quick-start)
4. [Core Functionalities](#3-core-functionalities)
5. [Important APIs at a Glance](#4-important-apis-at-a-glance)
6. [Data I/O](#5-data-io)
7. [Data Cleaning and Preprocessing](#6-data-cleaning-and-preprocessing)
8. [Performance Tips](#7-performance-tips)
9. [Learn More and Project Ideas](#8-learn-more-and-project-ideas)
10. [Glossary](#glossary)
11. [Common Gotchas](#common-gotchas)

---

## Prerequisites

This guide assumes you already have:

- **Python basics**: variables, functions, loops, lists, dictionaries, and importing modules.
- Some programming background (any language is fine).
- **Python 3.9+** and the ability to use `pip` and a terminal or Jupyter Notebook.
- *Helpful but optional:* basic familiarity with **NumPy** arrays. Pandas is built on top of NumPy.

---

## 1. What is pandas?

**pandas** is an open-source library for **data manipulation and analysis**. It gives you fast, flexible data structures for working with labeled, tabular, and time-series data, think of it as "Excel + SQL, but programmable."

### Core Concepts

| Concept | Description | Analogy |
|---|---|---|
| **Series** | A 1-D labeled array of values (single type) | One column in a spreadsheet |
| **DataFrame** | A 2-D labeled table with columns of possibly different types | A whole spreadsheet or SQL table |
| **Index** | The labels for rows (and a similar object for columns) | Row IDs / primary key |

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"])   # Series
df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})        # DataFrame

print(s.index)      # Index(['a', 'b', 'c'], dtype='object')
print(df.columns)   # Index(['x', 'y'], dtype='object')
```

### Typical Use Cases

- Loading and exploring datasets (CSV, Excel, JSON, SQL)
- Cleaning messy real-world data
- Summarizing and grouping data (reports, dashboards)
- Feature engineering for machine learning
- Time-series analysis (finance, sensors, logs)

---

## 2. Quick Start

### Installation

```bash
pip install pandas

# Optional: needed for Excel files
pip install openpyxl
```

Verify the installation:

```python
import pandas as pd
print(pd.__version__)
```

> **Convention:** always import pandas as `pd`.

### Minimal Example

```python
import pandas as pd

# 1. Create a DataFrame
data = {
    "name":   ["Sara", "Omar", "Laila", "Youssef"],
    "age":    [23, 31, 27, 35],
    "city":   ["Cairo", "Alexandria", "Cairo", "Giza"],
    "salary": [8000, 12000, 9500, 15000],
}
df = pd.DataFrame(data)

# 2. Inspect it
print(df.head())      # first rows
print(df.shape)       # (rows, columns) -> (4, 4)
print(df.dtypes)      # column types
df.info()             # summary: types, non-null counts, memory
print(df.describe())  # statistics for numeric columns

# 3. Basic operations
print(df["salary"].mean())              # average salary
print(df[df["age"] > 25])               # filter rows
df["salary_k"] = df["salary"] / 1000    # new column
print(df.sort_values("salary", ascending=False))
```

### Loading Data from a CSV

```python
df = pd.read_csv("data/students.csv")

df.head()        # first 5 rows
df.tail(3)       # last 3 rows
df.info()        # structure overview
```

---

## 3. Core Functionalities

### 3.1 Data Alignment

Pandas **aligns data by index labels** automatically during operations. Labels that don't match produce `NaN` rather than silently mismatching values.

```python
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["y", "z", "w"])

print(a + b)
# w     NaN
# x     NaN
# y    12.0
# z    23.0
# dtype: float64

print(a.add(b, fill_value=0))   # treat missing labels as 0
```

### 3.2 Selection and Filtering

```python
# Select columns
df["name"]                  # one column -> Series
df[["name", "salary"]]      # several columns -> DataFrame

# Label-based: .loc[rows, columns]
df.loc[0, "name"]
df.loc[0:2, ["name", "age"]]      # NOTE: label slices include the end

# Position-based: .iloc[rows, columns]
df.iloc[0, 1]
df.iloc[0:2, 0:2]                 # end is excluded (like Python lists)

# Boolean filtering
df[df["city"] == "Cairo"]
df[(df["age"] > 25) & (df["salary"] > 9000)]   # use & | ~, with parentheses
df[df["city"].isin(["Cairo", "Giza"])]

# query() gives a readable alternative
df.query("age > 25 and city == 'Cairo'")
```

### 3.3 Aggregation

```python
df["salary"].sum()
df["salary"].mean()
df["salary"].agg(["min", "max", "median", "std"])

df.agg({"age": "mean", "salary": ["min", "max"]})
```

### 3.4 Grouping (Split → Apply → Combine)

```python
# Average salary per city
df.groupby("city")["salary"].mean()

# Multiple aggregations with named outputs
summary = df.groupby("city").agg(
    avg_salary=("salary", "mean"),
    max_age=("age", "max"),
    headcount=("name", "count"),
)
print(summary)

# Group size, and keeping the group key as a normal column
df.groupby("city").size()
df.groupby("city", as_index=False)["salary"].sum()
```

### 3.5 Handling Missing Data

Missing values appear as `NaN` (or `None`/`pd.NA`).

```python
import numpy as np

df2 = pd.DataFrame({"a": [1, np.nan, 3], "b": ["x", None, "z"]})

df2.isna()                 # True where values are missing
df2.isna().sum()           # count of missing values per column
df2.dropna()               # drop rows with any missing value
df2.dropna(subset=["a"])   # only consider column 'a'
df2.fillna({"a": 0, "b": "unknown"})
df2["a"].fillna(df2["a"].mean())   # fill with the mean
```

---

## 4. Important APIs at a Glance

### Series

```python
s = pd.Series([3, 1, 2, 3], name="scores")

s.head()                  # first rows
s.unique()                # array([3, 1, 2])
s.nunique()               # 3
s.value_counts()          # frequency of each value
s.sort_values()           # sort by values
s.map({1: "low", 2: "mid", 3: "high"})   # element-wise mapping
s.apply(lambda v: v ** 2)                # custom function (slower)
s.astype("float64")
s.between(2, 3)           # boolean mask
```

### DataFrame

```python
df.columns                         # column labels
df.rename(columns={"name": "full_name"})
df.drop(columns=["salary_k"])      # remove columns
df.drop(index=[0, 1])              # remove rows
df.set_index("name")               # use a column as the index
df.reset_index(drop=True)          # restore a default 0..n-1 index
df.sort_values(["city", "salary"], ascending=[True, False])
df.assign(bonus=lambda d: d["salary"] * 0.1)   # add columns (chainable)
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 100],
                         labels=["young", "mid", "senior"])
df.nlargest(2, "salary")
df.sample(2, random_state=42)      # random rows (reproducible)
```

### Combining DataFrames

```python
# Stack vertically
pd.concat([df1, df2], ignore_index=True)

# SQL-style join
pd.merge(employees, departments, on="dept_id", how="left")
```

### Reshaping

```python
# Pivot table: rows = city, columns = age_group, values = mean salary
df.pivot_table(index="city", columns="age_group",
               values="salary", aggfunc="mean", observed=True)

# Wide -> long
pd.melt(wide_df, id_vars=["id"], var_name="metric", value_name="value")
```

---

## 5. Data I/O

### CSV

```python
df = pd.read_csv(
    "data.csv",
    sep=",",                      # delimiter
    usecols=["name", "salary"],   # read only needed columns
    dtype={"name": "string"},     # set types up front
    parse_dates=["hire_date"],    # parse dates (if the column exists)
    na_values=["N/A", "-"],       # extra missing-value markers
    encoding="utf-8",             # e.g. for Arabic text
)

df.to_csv("output.csv", index=False)   # index=False avoids an extra column
```

### Excel

```python
# Requires: pip install openpyxl
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Write multiple sheets
with pd.ExcelWriter("report.xlsx") as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    summary.to_excel(writer, sheet_name="Summary")
```

### JSON

```python
df = pd.read_json("data.json")

df.to_json("output.json", orient="records", indent=2, force_ascii=False)
# orient="records" -> [{"col": value, ...}, ...]

# For nested JSON, flatten it first
from pandas import json_normalize
flat = json_normalize(nested_data, sep="_")
```

> 💡 Pandas also reads Parquet (`read_parquet`), SQL (`read_sql`), and more.

---

## 6. Data Cleaning and Preprocessing

A typical cleaning workflow:

```python
import pandas as pd

df = pd.read_csv("messy_data.csv")

# 1. Inspect
df.info()
print(df.isna().sum())
print(df.duplicated().sum())

# 2. Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df = df.dropna(subset=["email"])          # email is required

# 4. Fix data types
df["price"] = pd.to_numeric(df["price"], errors="coerce")   # bad values -> NaN
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["category"] = df["category"].astype("category")
df["age"] = df["age"].astype("int64")     # only after removing NaN

# 5. Clean text
df["name"] = df["name"].str.strip().str.title()

# 6. Remove duplicates
df = df.drop_duplicates()                             # fully identical rows
df = df.drop_duplicates(subset=["email"], keep="first")

# 7. Sanity-check the result
df.info()
df.describe()
```

---

## 7. Performance Tips

### Prefer vectorization over loops

Vectorized operations run in optimized C code; Python loops do not.

```python
# ❌ Slow: row-by-row loop
totals = []
for i in range(len(df)):
    totals.append(df.loc[i, "price"] * df.loc[i, "qty"])
df["total"] = totals

# ✅ Fast: vectorized
df["total"] = df["price"] * df["qty"]

# ✅ Conditional logic without loops
import numpy as np
df["tier"] = np.where(df["total"] > 1000, "high", "low")
```

**Rule of thumb:** `vectorized operation` > `.str` / `.dt` accessors > `.apply()` > `iterrows()`.

### Other tips

- **Use efficient dtypes:** convert low-cardinality text columns to `category`, and downcast numbers when possible.
  ```python
  df["city"] = df["city"].astype("category")
  df.memory_usage(deep=True)
  ```
- **Load only what you need:** `usecols=[...]`, `nrows=...`.
- **Process large files in chunks:**
  ```python
  total = 0
  for chunk in pd.read_csv("big.csv", chunksize=100_000):
      total += chunk["amount"].sum()
  ```
- **Use `inplace` thoughtfully:** `inplace=True` (e.g., `df.drop(..., inplace=True)`) modifies the object directly, but it usually **does not** save memory or time, since pandas often copies internally. Method chaining with assignment is the recommended style:
  ```python
  df = (
      df.dropna(subset=["price"])
        .drop_duplicates()
        .assign(total=lambda d: d["price"] * d["qty"])
  )
  ```
  Use `inplace=True` only for quick, interactive work where it makes the code simpler.

---

## 8. Learn More and Project Ideas

### Recommended Resources

- 📘 [Official pandas documentation](https://pandas.pydata.org/docs/), including the *10 Minutes to pandas* tutorial and the *User Guide*
- 📗 [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) (official PDF)
- 📙 *Python for Data Analysis* by Wes McKinney (creator of pandas)
- 🎓 [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas): short, interactive micro-course
- 🧪 [Kaggle Datasets](https://www.kaggle.com/datasets) and [UCI ML Repository](https://archive.ics.uci.edu/): practice data

### Project Ideas

| Level | Project | Skills Practiced |
|---|---|---|
| 🟢 Beginner | Analyze the Titanic dataset: survival rate by class, age, and gender | Filtering, `groupby`, missing values |
| 🟢 Beginner | Personal expense tracker from a CSV export | Dates, aggregation, `pivot_table` |
| 🟡 Intermediate | Clean and analyze a messy survey dataset | Cleaning, type conversion, duplicates |
| 🟡 Intermediate | Sales dashboard data: monthly revenue by region and product | `merge`, `groupby`, time-series resampling |
| 🔴 Advanced | Feature engineering pipeline for an ML model (e.g., house prices) | Encoding, scaling, vectorization, `pipe` |
| 🔴 Advanced | Analyze a large log/sensor file using chunked processing | Performance, `chunksize`, dtypes |

---

## Glossary

| Term | Meaning |
|---|---|
| **Series** | One-dimensional labeled array |
| **DataFrame** | Two-dimensional labeled table |
| **Index** | Labels identifying rows (or columns) |
| **dtype** | Data type of a column (`int64`, `float64`, `object`, `string`, `category`, `datetime64`) |
| **NaN / NA** | "Not a Number" / missing value marker |
| **Axis** | Direction of an operation: `axis=0` (rows, down) or `axis=1` (columns, across) |
| **Vectorization** | Applying an operation to a whole array at once instead of looping |
| **Boolean mask** | A True/False Series used to filter rows |
| **GroupBy** | Splitting data into groups, applying a function, and combining results |
| **Merge / Join** | Combining DataFrames based on matching keys |
| **Pivot table** | A summary table reshaped by two categorical dimensions |
| **Method chaining** | Linking several operations in one expression |
| **View vs. copy** | Whether a result shares memory with the original or is independent |

---
