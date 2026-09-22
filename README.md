# 🚀 Machine Learning Course Overview

### A Complete Path from Python to Classical Machine Learning

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Level](https://img.shields.io/badge/level-beginner--to--intermediate-blue)
![Language](https://img.shields.io/badge/language-Python-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> A structured, project-driven roadmap that takes you from Python fundamentals all the way to a full classical Machine Learning pipeline — with math foundations, hands-on projects, and review quizzes built into every module.

---

## 📖 About This Course

This repository lays out a **6-module curriculum** designed to build Machine Learning skills from the ground up. Each module combines theory, applied coding, and the exact math needed to understand *why* each technique works — not just how to call it.

Throughout the roadmap you'll find three recurring markers:

| Marker | Meaning |
|:---:|---|
| **Σ** | **Math in Focus** — the exact moment a math concept is introduced, tied directly to the technique being taught |
| **★** | **Milestone Project** — a hands-on, portfolio-worthy project applying everything learned so far |
| **✎** | **Review Quiz** — an end-of-module quiz to reinforce and test understanding |

---

## 🗂️ Table of Contents

- [Module 1 — Python Programming Foundations](#module-1--python-programming-foundations)
- [Module 2 — Understanding Data](#module-2--understanding-data)
- [Module 3 — Data Preprocessing](#module-3--data-preprocessing)
- [Module 4 — Exploratory Data Analysis (EDA)](#module-4--exploratory-data-analysis-eda)
- [Module 5 — Introduction to Machine Learning](#module-5--introduction-to-machine-learning)
- [Module 6 — Classical Machine Learning](#module-6--classical-machine-learning)
- [Projects Overview](#-projects-overview)
- [Prerequisites](#-prerequisites)
- [How to Use This Roadmap](#-how-to-use-this-roadmap)

---

## Module 1 — Python Programming Foundations

The base layer of the roadmap: core Python, environment setup, and the two libraries every ML workflow depends on — **NumPy** and **Pandas** — plus data visualization.

<details>
<summary><strong>Click to expand topics</strong></summary>

- Environment setup (Jupyter Notebook / Google Colab)
- Virtual environments (`venv` & `conda`)
- Variables, data types, operators
- Conditionals, loops, loop control (`break`, `continue`, `pass`)
- Functions, parameters, return values, scope
- Lists, tuples, dictionaries, sets, comprehensions
- String manipulation & file I/O
- Error & exception handling
- **Object-Oriented Programming**
  - Classes, objects, `__init__`, `self`
  - **OOP Core Concepts:** Encapsulation, Inheritance, Polymorphism, Abstraction (`abc` module)
- Modules, packages & `pip`
- **NumPy:** array creation & indexing, vectorized operations, broadcasting, aggregations, boolean masking
  - Σ *Vectors & matrices, matrix operations*
- **Pandas:** Series vs. DataFrame, filtering, indexing (`loc`/`iloc`), `groupby`, `pivot_table`, `merge`
- **Matplotlib & Seaborn:** line/bar/scatter plots, styling, statistical plots

✎ **End-of-Module 1 Review Quiz**

</details>

---

## Module 2 — Understanding Data

Before modeling anything, you need to understand what you're modeling.

<details>
<summary><strong>Click to expand topics</strong></summary>

- Structured vs. unstructured data
- Qualitative vs. quantitative data
- Variable types: numerical (discrete/continuous), categorical, ordinal
- Data sources: files, APIs, web scraping, sensors
- Loading data with Pandas (CSV, Excel, JSON, SQL)
- Dataset structure inspection (`.info()`, `.describe()`, shape, dtypes)
- Descriptive statistics
  - Σ *Mean, median, mode, variance, standard deviation*

✎ **End-of-Module 2 Review Quiz**

</details>

---

## Module 3 — Data Preprocessing

Raw data is messy. This module covers cleaning, transforming, and preparing data for modeling.

<details>
<summary><strong>Click to expand topics</strong></summary>

- Common data quality issues
- Handling missing values: deletion & imputation (mean/median/mode/model-based)
- Duplicate detection & removal
- Outlier detection & treatment
  - Σ *Z-score method, Interquartile Range (IQR)*
- Data type conversion & date parsing
- **Encoding:** Label, One-Hot, Ordinal
- **Feature Scaling:** Normalization & Standardization
  - Σ *Min-max scaling formula, Z-score scaling formula*
- Handling imbalanced data (SMOTE, oversampling/undersampling)
- Feature engineering & feature selection (filter, wrapper, embedded methods)
- Train / Validation / Test splitting, stratified sampling
  - Σ *Basic probability & sampling theory*
- Building preprocessing pipelines with scikit-learn

✎ **End-of-Module 3 Review Quiz**

</details>

---

## Module 4 — Exploratory Data Analysis (EDA)

**EDA Flow:** `Data Overview → Univariate Analysis → Bivariate & Multivariate Analysis → Identifying Patterns → Insights & Hypotheses`

<details>
<summary><strong>Click to expand topics</strong></summary>

- Purpose & process of EDA
- Descriptive statistics recap: percentiles, five-number summary
- **Univariate Analysis:** histograms, box plots
  - Σ *Skewness & kurtosis*
- **Probability Distributions:** discrete vs. continuous
  - Σ *Normal distribution, uniform distribution, Central Limit Theorem*
- **Bivariate Analysis:** scatter plots, cross-tabulation
- **Correlation & Covariance**
  - Σ *Correlation coefficient, covariance matrix*
- **Multivariate Analysis:** pair plots, correlation heatmaps
- Identifying patterns, trends & anomalies
- Turning insights into modeling hypotheses

★ **Middle Project 1 — Exploratory Data Analysis Project**
> Apply the full EDA flow to a real-world dataset: profile the data, analyze univariate/bivariate/multivariate patterns, and deliver a written summary of insights & hypotheses.

✎ **End-of-Module 4 Review Quiz**

</details>

---

## Module 5 — Introduction to Machine Learning

The conceptual bridge between data science and modeling.

<details>
<summary><strong>Click to expand topics</strong></summary>

- What is Machine Learning?
- AI vs. Machine Learning vs. Deep Learning
- Supervised, Unsupervised & Reinforcement Learning — core concepts
- The Machine Learning workflow (raw data → deployed model)
- **Math Foundations:** vectors, matrices, dot products
  - Σ *Vector/matrix notation, dot product*
- **Math Foundations:** derivatives & gradients (conceptual)
  - Σ *Derivatives, gradients*
- Features, labels & model representation
- Training, validation & testing — concepts

✎ **End-of-Module 5 Review Quiz**

</details>

---

## Module 6 — Classical Machine Learning

The core of the roadmap — every major classical ML algorithm, with the math and evaluation methodology behind each.

<details>
<summary><strong>Click to expand topics</strong></summary>

**Regression**
- Simple & Multiple Linear Regression
  - Σ *Cost function (MSE), gradient descent, normal equation*
- Training/validation/testing in practice
- Overfitting vs. underfitting
  - Σ *Bias-variance tradeoff*
- Polynomial Regression
- Regularization: Ridge (L2) & Lasso (L1)
- Regression evaluation metrics
  - Σ *MAE, MSE, RMSE, R²*

★ **Middle Project 2 — Regression Project**
> Build an end-to-end regression pipeline: preprocess, engineer features, fit & regularize a model, and report evaluation metrics.

**Classification**
- Logistic Regression (binary & multiclass)
  - Σ *Sigmoid function, log-loss*
- K-Nearest Neighbors (KNN)
  - Σ *Euclidean & Manhattan distance*
- Decision Trees
  - Σ *Entropy, information gain, Gini index*
- Support Vector Machines (SVM)
  - Σ *Hyperplanes, margins, kernel functions*
- Naive Bayes
  - Σ *Bayes' theorem, conditional independence*
- Classification evaluation metrics
  - Σ *Confusion matrix, precision, recall, F1, ROC-AUC*
- Ensemble Learning: Bagging, Random Forests, Boosting (Gradient Boosting, XGBoost/LightGBM)
- Cross-validation (k-fold)
- Hyperparameter tuning (grid search, random search)

★ **Middle Project 3 — Classification Project**
> Build an end-to-end classification pipeline: compare multiple classifiers, tune hyperparameters with cross-validation, and evaluate with the right metrics.

**Unsupervised Learning**
- K-Means Clustering
  - Σ *Centroids, within-cluster variance*
- Hierarchical Clustering
  - Σ *Linkage methods*
- DBSCAN
- Dimensionality Reduction: PCA
  - Σ *Eigenvalues, eigenvectors, explained variance*

★ **Middle Project 4 — Unsupervised Learning Project**
> Explore an unlabeled dataset: apply clustering to discover groups and PCA to reduce dimensionality, then interpret and visualize the results.

**Capstone**
- End-to-end classical ML project workflow

★ **Final Project — Capstone: Complete End-to-End ML Pipeline**
> Take a raw, real-world dataset all the way through data understanding, preprocessing, EDA, model selection (regression, classification, or clustering), tuning, evaluation, and a final results presentation.

✎ **End-of-Module 6 Review Quiz**

</details>

---

## 🏗️ Projects Overview

| Project | Module | Focus |
|---|---|---|
| Exploratory Data Analysis Project | 4 | Full EDA flow on a real dataset |
| Regression Project | 6 | End-to-end regression pipeline |
| Classification Project | 6 | Multi-classifier comparison + tuning |
| Unsupervised Learning Project | 6 | Clustering + PCA |
| **Capstone Project** | 6 | Complete end-to-end ML pipeline |

---

## ✅ Prerequisites

- No prior programming experience required — Module 1 starts from scratch
- Basic high-school level math is helpful but not mandatory (all required math is introduced in-context via the Σ markers)

---

## 🧭 How to Use This Roadmap

1. Progress module by module — each one builds on the last.
2. Don't skip the Σ math boxes — they're placed exactly where you need them, not as a separate math course.
3. Complete every ★ project before moving on — they're checkpoints, not extras.
4. Use the ✎ quizzes to confirm you're ready to advance.
5. By the end of Module 6, you'll have a full portfolio: an EDA project, a regression project, a classification project, an unsupervised learning project, and a capstone.

---

<p align="center">Happy learning! 📊🤖</p>
