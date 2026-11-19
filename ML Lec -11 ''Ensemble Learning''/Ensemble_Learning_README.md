# Ensemble Learning

### Complete Lecture Notes, Visual Explanations, Mathematics, and Python Examples

> Based on the 37-slide **Ensemble Learning** lecture by TriNode Team.\
> This README follows the lecture's structure and terminology, with
> explanations and runnable-style examples using Python and
> scikit-learn.

------------------------------------------------------------------------

## Table of Contents

1.  [Introduction to Ensemble
    Learning](#1-introduction-to-ensemble-learning)
2.  [Bias, Variance, and Ensembles](#2-bias-variance-and-ensembles)
3.  [Ways to Combine Models](#3-ways-to-combine-models)
4.  [Voting Ensembles](#4-voting-ensembles)
5.  [Bootstrap Sampling](#5-bootstrap-sampling)
6.  [Bagging](#6-bagging-bootstrap-aggregating)
7.  [Out-of-Bag Evaluation](#7-out-of-bag-oob-evaluation)
8.  [Random Forest](#8-random-forest)
9.  [Feature Importance](#9-feature-importance)
10. [AdaBoost](#10-adaboost)
11. [Gradient Boosting](#11-gradient-boosting)
12. [Learning Rate and Number of
    Trees](#12-learning-rate-and-number-of-trees)
13. [Bagging vs. Boosting](#13-bagging-vs-boosting)
14. [Boosting at Scale: XGBoost and
    LightGBM](#14-boosting-at-scale-xgboost-and-lightgbm)
15. [Histogram Gradient Boosting](#15-histogram-gradient-boosting)
16. [Early Stopping](#16-early-stopping)
17. [Important Equations](#17-important-equations)
18. [Quick Reference](#18-quick-reference)
19. [Practice Questions](#19-practice-questions)

------------------------------------------------------------------------

# 1. Introduction to Ensemble Learning

## 1.1 What is Ensemble Learning?

**Ensemble Learning** is a machine learning technique that combines the
predictions of multiple models to produce one final prediction.

Instead of relying on a single model, we train several models and
combine their outputs.

The main idea is:

> Many models together can perform better than one model alone.

### A real-world analogy

Imagine asking five people to estimate the weight of an object. Their
estimates may be:

-   Person 1: 48 kg
-   Person 2: 52 kg
-   Person 3: 50 kg
-   Person 4: 55 kg
-   Person 5: 49 kg

The average is 50.8 kg. Combining independent estimates can produce a
more reliable answer than relying on one estimate.

The same idea is used in machine learning.

## 1.2 How are predictions combined?

-   **Classification:** combine class predictions using voting, or
    combine class probabilities.
-   **Regression:** average the numerical predictions.

### Classification example: majority voting

``` text
Model 1  -> Cat
Model 2  -> Dog
Model 3  -> Cat

Final prediction -> Cat
```

Cat receives two votes, while Dog receives one.

### Regression example: averaging

Suppose three models predict a house price:

``` text
Model 1 -> 1,000,000
Model 2 -> 1,100,000
Model 3 ->   900,000
```

The ensemble prediction is:

\[ `\hat `{=tex}y = `\frac{1,000,000 + 1,100,000 + 900,000}{3}`{=tex} =
1,000,000 \]

## 1.3 Conditions for a useful ensemble

The lecture highlights two important conditions:

1.  The models should perform better than random guessing.
2.  The models should make different errors.

If every model is an identical copy and makes exactly the same mistakes,
combining them adds little or no value.

The diversity of errors is important because averaging or voting can
cancel out some individual model errors.

### Visual: majority voting

``` mermaid
flowchart TD
    A[Input sample] --> B[Model 1: Cat]
    A --> C[Model 2: Dog]
    A --> D[Model 3: Cat]
    B --> E[Majority vote]
    C --> E
    D --> E
    E --> F[Final prediction: Cat]
```

## 1.4 Where is Ensemble Learning used?

Common ensemble algorithms include:

-   Random Forest
-   AdaBoost
-   Gradient Boosting
-   XGBoost
-   LightGBM

These methods are especially common in tabular-data problems.

------------------------------------------------------------------------

# 2. Bias, Variance, and Ensembles

Understanding **Bias** and **Variance** explains why Bagging and
Boosting work.

## 2.1 Bias

**Bias** is error caused by a model that is too simple to learn the
underlying pattern in the data.

A high-bias model tends to underfit.

Example:

-   The real relationship between input and output is curved.
-   We use a simple linear model.
-   A straight line cannot represent the curved relationship well.

This is called **Underfitting**.

## 2.2 Variance

**Variance** describes how sensitive a model is to changes in its
training data.

A high-variance model may change significantly when the training data
changes slightly.

A very deep decision tree may learn noise and small details in the
training set. It can perform very well on training data but poorly on
unseen data.

This is called **Overfitting**.

## 2.3 Bias vs. Variance

  -----------------------------------------------------------------------
  Concept                 High Bias               High Variance
  ----------------------- ----------------------- -----------------------
  Main cause              Model is too simple     Model is too sensitive

  Typical problem         Underfitting            Overfitting

  Training behavior       Misses important        May memorize
                          patterns                details/noise

  Common example          Very simple model for a Very deep decision tree
                          complex pattern         
  -----------------------------------------------------------------------

## 2.4 Which ensemble addresses which problem?

The lecture's central comparison is:

-   **Bagging** averages many high-variance models, often deep trees. It
    mainly reduces variance.
-   **Boosting** adds many high-bias models, often shallow trees, one
    after another. It mainly reduces bias.

``` mermaid
flowchart TD
    A[Ensemble methods] --> B[Bagging]
    A --> C[Boosting]
    B --> D[Many high-variance models]
    D --> E[Train independently / in parallel]
    E --> F[Aggregate predictions]
    F --> G[Main effect: reduce variance]
    C --> H[Many high-bias weak learners]
    H --> I[Train sequentially]
    I --> J[Each learner improves the ensemble]
    J --> K[Main effect: reduce bias]
```

------------------------------------------------------------------------

# 3. Ways to Combine Models

The lecture introduces four major approaches.

## 3.1 Voting

Train different types of models on the same training data, then combine
their predictions.

Examples of different model types:

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Decision Tree

Voting can be:

-   **Hard voting:** use the majority of predicted class labels.
-   **Soft voting:** average predicted class probabilities.

## 3.2 Bagging

Train the same type of model on different bootstrap samples, then
combine the predictions.

Example: train many decision trees, each on a different bootstrap
sample.

## 3.3 Boosting

Train models one after another. Each new model focuses on what the
previous ensemble still gets wrong.

## 3.4 Stacking

Train several base models, then feed their predictions into a final
model called a **meta-model**. The meta-model learns how to combine the
base predictions.

### Comparison

  Method     Main idea
  ---------- ------------------------------------------------------
  Voting     Different models vote or average probabilities
  Bagging    Same model type, different bootstrap samples
  Boosting   Sequential models improve the current ensemble
  Stacking   A meta-model learns how to combine model predictions

### Common scikit-learn classes

  Method          Class
  --------------- ----------------------------------------------------
  Voting          `VotingClassifier`
  Bagging         `BaggingClassifier`
  Random Forest   `RandomForestClassifier`
  Boosting        `AdaBoostClassifier`, `GradientBoostingClassifier`
  Stacking        `StackingClassifier`

Regressor versions are also available for regression tasks.

------------------------------------------------------------------------

# 4. Voting Ensembles

## 4.1 Hard Voting

Each model predicts a class label. The class with the most votes wins.

Example:

``` text
Logistic Regression -> 1
KNN                 -> 0
Decision Tree       -> 1

Final class         -> 1
```

## 4.2 Soft Voting

Each model predicts a probability for every class. The ensemble averages
those probabilities and selects the class with the highest average
probability.

For a binary problem, suppose the models predict the probability of
class 1:

  Model       Probability of class 1
  --------- ------------------------
  Model 1                       0.80
  Model 2                       0.60
  Model 3                       0.40

\[ P(`\text{class 1}`{=tex}) = `\frac{0.80 + 0.60 + 0.40}{3}`{=tex} =
0.60 \]

The average probability is 0.60.

## 4.3 Voting example with scikit-learn

The lecture uses the Breast Cancer dataset and three models:

1.  Logistic Regression
2.  KNN
3.  Decision Tree

Logistic Regression and KNN are placed in pipelines with
`StandardScaler`, because they are sensitive to feature scales. Decision
Trees generally do not require feature scaling.

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split: 75% training, 25% testing
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

# Scale models that are sensitive to feature scale
scaled = lambda model: make_pipeline(
    StandardScaler(),
    model
)

models = [
    ("logistic", scaled(LogisticRegression(max_iter=1000))),
    ("knn", scaled(KNeighborsClassifier())),
    ("tree", DecisionTreeClassifier(random_state=0))
]

# Evaluate individual models
for name, model in models:
    model.fit(X_tr, y_tr)
    accuracy = model.score(X_te, y_te)
    print(f"{name:9} {accuracy:.3f}")

# Soft voting ensemble
vote = VotingClassifier(
    estimators=models,
    voting="soft"
)

vote.fit(X_tr, y_tr)
print(f"vote      {vote.score(X_te, y_te):.3f}")
```

### Results reported in the lecture

  Model                   Accuracy
  --------------------- ----------
  Logistic Regression        0.986
  KNN                        0.979
  Decision Tree              0.937
  Soft Voting                0.958

**Important observation:** an ensemble is not guaranteed to outperform
every individual model on every dataset. In this lecture's experiment,
Soft Voting is better than the Decision Tree but lower than Logistic
Regression and KNN.

------------------------------------------------------------------------

# 5. Bootstrap Sampling

## 5.1 What is Bootstrap?

**Bootstrap sampling** creates a new sample from the original training
data by randomly drawing rows **with replacement**.

With replacement means that after a row is selected, it is returned to
the pool and may be selected again.

Suppose the original dataset contains:

``` text
A, B, C, D, E
```

One possible bootstrap sample of size five is:

``` text
A, C, C, E, A
```

Notice:

-   A appears twice.
-   C appears twice.
-   B and D do not appear.
-   The sample still contains five draws.

## 5.2 Why do we use Bootstrap?

Different bootstrap samples contain different combinations of training
rows. Models trained on these samples can therefore become different
from one another.

This diversity is useful when their predictions are aggregated.

## 5.3 Probability that a row is selected

Suppose the original dataset has (n) rows, and we draw (n) times with
replacement.

The probability that one particular row is missed in a single draw is:

\[ 1-`\frac{1}{n}`{=tex} \]

The probability that it is missed in all (n) independent draws is:

\[ `\left`{=tex}(1-`\frac{1}{n}`{=tex}`\right`{=tex})\^n \]

Therefore, the probability that it is selected at least once is:

\[ P(`\text{picked}`{=tex}) =
1-`\left`{=tex}(1-`\frac{1}{n}`{=tex}`\right`{=tex})\^n \]

For large (n):

\[ P(`\text{picked}`{=tex}) `\approx 1`{=tex}-`\frac{1}{e}`{=tex}
`\approx 0.632`{=tex} \]

So a bootstrap sample contains approximately:

-   **63.2%** distinct original rows.
-   **36.8%** rows that were not selected.

The unselected rows are called **Out-of-Bag (OOB)** rows for that
sample.

## 5.4 Bootstrap code

``` python
import numpy as np

rng = np.random.default_rng(0)

# Draw 10 values from 0..9, with replacement
sample = rng.choice(
    10,
    size=10,
    replace=True
)

print("bootstrap sample:", np.sort(sample))

# Find values that were not selected
out_of_bag = np.setdiff1d(
    np.arange(10),
    sample
)

print("out-of-bag:", out_of_bag)
```

### Measure the distinct-row percentage

``` python
for n in [10, 100, 1_000, 100_000]:
    sample = rng.choice(
        n,
        size=n,
        replace=True
    )

    share = len(np.unique(sample)) / n
    print(f"n = {n:<7} distinct share = {share:.3f}")

print("Theoretical limit:", round(1 - np.exp(-1), 3))
```

As the dataset size increases, the observed distinct-row percentage
approaches approximately 0.632.

------------------------------------------------------------------------

# 6. Bagging = Bootstrap Aggregating

## 6.1 What is Bagging?

Bagging combines Bootstrap sampling and Aggregation.

It follows three main steps:

1.  Create multiple bootstrap samples from the training data.
2.  Train one model on each bootstrap sample.
3.  Aggregate the predictions of all trained models.

The models can be trained in parallel because each model is trained
independently.

### Workflow

``` mermaid
flowchart TD
    A[Original training dataset] --> B[Bootstrap sampling]
    B --> C1[Bootstrap sample 1]
    B --> C2[Bootstrap sample 2]
    B --> C3[Bootstrap sample B]
    C1 --> D1[Train model 1]
    C2 --> D2[Train model 2]
    C3 --> D3[Train model B]
    D1 --> E[Aggregate predictions]
    D2 --> E
    D3 --> E
    E --> F[Final prediction]
```

## 6.2 How are predictions aggregated?

For classification:

-   Majority voting, or
-   Averaging predicted probabilities.

For regression:

-   Average the numerical predictions.

The bagging prediction for regression can be written as:

\[
f\_{`\text{bag}`{=tex}}(x)=`\frac{1}{B}`{=tex}`\sum`{=tex}\_{b=1}\^{B}f_b(x)
\]

Where:

-   (B): number of models.
-   (f_b(x)): prediction from model (b).
-   (f\_{`\text{bag}`{=tex}}(x)): final ensemble prediction.

## 6.3 Why does Bagging reduce Variance?

A deep decision tree can be sensitive to the training sample. Different
bootstrap samples produce trees that make different errors.

Averaging their predictions can cancel some of that variation.

Bagging mainly reduces variance. The lecture describes the base models
as high-variance models, often deep decision trees.

## 6.4 Variance formula

The lecture gives the variance of an average of (B) models as:

\[ Var = `\rho`{=tex}`\sigma`{=tex}\^2 +
`\frac{(1-\rho)\sigma^2}{B}`{=tex} \]

Where:

-   (`\rho`{=tex}): correlation between models.
-   (`\sigma`{=tex}\^2): variance of one model.
-   (B): number of models.

Example:

\[
`\sigma`{=tex}\^2=1,`\quad `{=tex}`\rho=0.5`{=tex},`\quad `{=tex}B=100
\]

\[ Var=0.5+`\frac{0.5}{100}`{=tex}=0.505 \]

The variance is reduced from 1 to 0.505 in this example.

The formula also shows why model diversity matters: lower correlation
can make averaging more effective.

## 6.5 Bagging code

``` python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

for n in [1, 5, 25, 100, 300]:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n,
        random_state=0
    )

    bag.fit(X_tr, y_tr)
    accuracy = bag.score(X_te, y_te)

    print(f"{n:3} trees: test accuracy = {accuracy:.3f}")
```

### Results reported in the lecture

The lecture uses a noisy-moons dataset with 280 training rows and 120
test rows.

    Number of trees   Test accuracy
  ----------------- ---------------
                  1           0.825
                  5           0.867
                 25           0.908
                100           0.892
                300           0.900

The score does not have to increase at every step. The main pattern is
that combining trees can improve generalization compared with one tree.

------------------------------------------------------------------------

# 7. Out-of-Bag (OOB) Evaluation

## 7.1 What is OOB?

Each bootstrap sample contains approximately 63.2% of the distinct
training rows. The remaining approximately 36.8% are not used to train
that particular model.

Those excluded rows are its **Out-of-Bag samples**.

## 7.2 How does OOB evaluation work?

For each training row:

1.  Identify the trees whose bootstrap samples did not include that row.
2.  Let only those trees predict the row.
3.  Aggregate their predictions.
4.  Compare the result with the true label.

The resulting score is called the **OOB score**.

It estimates generalization performance without requiring a separate
validation split for this purpose.

## 7.3 OOB code

``` python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    oob_score=True,
    random_state=0,
    n_jobs=-1
)

rf.fit(X_tr, y_tr)

print("Test accuracy:", rf.score(X_te, y_te))
print("OOB accuracy:", rf.oob_score_)
```

### When is OOB useful?

-   Small datasets, where keeping aside a validation set may be costly.
-   Quick checks while tuning the number of trees.
-   A sanity check alongside cross-validation.

**Note:** OOB is not a replacement for an independent final test set.

------------------------------------------------------------------------

# 8. Random Forest

## 8.1 What is Random Forest?

Random Forest combines:

**Bagging + Random Feature Selection**

It trains multiple decision trees on bootstrap samples. At every split,
each tree considers only a random subset of the available features.

## 8.2 Why is random feature selection needed?

Imagine one feature is extremely strong. In ordinary bagging, many trees
may choose that same feature for their first split.

The trees may then become similar and make correlated errors.

Random Forest reduces this problem by allowing each split to consider
only a random subset of features.

### Visual: feature selection at a split

``` mermaid
flowchart TD
    A[All available features: p] --> B[Randomly select a subset]
    B --> C[Candidate features]
    C --> D[Choose the best split among candidates]
    D --> E[Continue growing the tree]
```

## 8.3 How many features are considered?

The lecture gives the common classification setting:

\[ max_features `\approx `{=tex}`\sqrt{p}`{=tex} \]

Where (p) is the total number of features.

For the Breast Cancer dataset:

\[ p=30 \]

\[ `\sqrt{30}`{=tex}`\approx5.48`{=tex} \]

So approximately five features are considered at each split in the
lecture's example.

The lecture notes that scikit-learn commonly considers all features for
regression by default.

## 8.4 Bagging vs. Random Forest

  -----------------------------------------------------------------------
  Property                Bagging                 Random Forest
  ----------------------- ----------------------- -----------------------
  Rows per tree           Bootstrap sample        Bootstrap sample

  Features per split      All features            Random subset

  Tree correlation        Often higher            Often lower

  Main benefit            Reduce variance         Reduce variance further
                                                  through decorrelation
  -----------------------------------------------------------------------

## 8.5 Why do random features help?

Recall:

\[ Var(forest)=`\rho`{=tex}`\sigma`{=tex}\^2+
`\frac{(1-\rho)\sigma^2}{B}`{=tex} \]

Adding trees makes the second term smaller. However, the first term,
(`\rho`{=tex}`\sigma`{=tex}\^2), remains.

Random feature selection can lower the correlation (`\rho`{=tex}),
reducing the variance floor.

Example from the lecture:

\[ `\sigma`{=tex}\^2=1,`\quad `{=tex}B=500 \]

If (`\rho=0.6`{=tex}):

\[ Var=0.6+`\frac{0.4}{500}`{=tex}=0.6008 \]

If (`\rho=0.2`{=tex}):

\[ Var=0.2+`\frac{0.8}{500}`{=tex}=0.2016 \]

The individual trees may become slightly weaker because they cannot
always use the strongest feature, but the forest can become better
because the trees are less correlated.

## 8.6 Random Forest code

``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# One decision tree
tree = DecisionTreeClassifier(
    random_state=0
)
tree.fit(X_tr, y_tr)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    oob_score=True,
    random_state=0,
    n_jobs=-1
)

rf.fit(X_tr, y_tr)

print("Tree test accuracy:", tree.score(X_te, y_te))
print("Forest test accuracy:", rf.score(X_te, y_te))
print("Forest OOB accuracy:", rf.oob_score_)

# Number of features considered at each split
print("Features per split:", rf.estimators_[0].max_features_)
```

### Results reported in the lecture

  Model                 Test accuracy
  ------------------- ---------------
  Decision Tree                 0.937
  Random Forest                 0.958
  Random Forest OOB             0.958

The lecture reports five features considered per split in this
experiment.

## 8.7 How many trees should we use?

The lecture uses `warm_start=True` to add trees while keeping the trees
already trained.

``` python
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

tree_counts = range(10, 401, 10)
oob_errors = []

rf = RandomForestClassifier(
    oob_score=True,
    warm_start=True,
    random_state=0
)

for n in tree_counts:
    rf.n_estimators = n
    rf.fit(X_tr, y_tr)
    oob_errors.append(1 - rf.oob_score_)

plt.plot(list(tree_counts), oob_errors, "o-")
plt.xlabel("Number of trees")
plt.ylabel("OOB error")
plt.title("Random Forest: OOB Error vs. Number of Trees")
plt.show()
```

`warm_start=True` keeps the existing trees and adds more when
`n_estimators` is increased.

The lecture's observations:

-   OOB error decreases at first.
-   It eventually flattens.
-   More trees usually add computational cost after the score has
    stabilized.

------------------------------------------------------------------------

# 9. Feature Importance

Random Forest can provide information about which features the model
relied on.

The lecture introduces two approaches.

## 9.1 Impurity-based importance

A tree split reduces impurity. Impurity-based importance adds up the
impurity reductions contributed by a feature across the trees.

In scikit-learn:

``` python
rf.feature_importances_
```

The importances sum to 1.

### Limitations

-   It can favor features with many possible split points.
-   It is measured using training data.
-   Correlated features can share importance, making each appear less
    important.

## 9.2 Permutation importance

Permutation importance measures how much model performance decreases
when one feature is shuffled.

Steps:

1.  Measure the model's baseline score.
2.  Shuffle one feature in the evaluation data.
3.  Measure the score again.
4.  A larger score drop indicates greater reliance on that feature.

``` python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf,
    X_te,
    y_te,
    n_repeats=10,
    random_state=0
)

for index in result.importances_mean.argsort()[::-1][:10]:
    print(
        data.feature_names[index],
        result.importances_mean[index]
    )
```

Permutation importance is useful because it evaluates the effect of a
feature on held-out data. Correlated features can still complicate
interpretation.

## 9.3 Plot impurity-based importance

``` python
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(
    rf.feature_importances_,
    index=data.feature_names
)

importance.sort_values().tail(10).plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Random Forest Features")
plt.show()
```

------------------------------------------------------------------------

# 10. AdaBoost

## 10.1 What is AdaBoost?

AdaBoost means **Adaptive Boosting**.

It trains weak learners sequentially. After each learner, it increases
the weights of misclassified training examples so that the next learner
focuses more on those difficult examples.

The lecture uses decision stumps as weak learners.

A **decision stump** is a decision tree with one split and two leaves.

## 10.2 AdaBoost workflow

``` mermaid
flowchart TD
    A[Training examples: equal weights] --> B[Train decision stump]
    B --> C[Calculate weighted error]
    C --> D[Calculate stump weight]
    D --> E[Increase weights of misclassified examples]
    E --> F[Train next stump]
    F --> G[Repeat]
    G --> H[Weighted final vote]
```

### Step 1: Start with equal weights

At the beginning, every training example has the same weight.

For 100 examples, each initial weight is:

\[ w_i=`\frac{1}{100}`{=tex}=0.01 \]

### Step 2: Train a stump

Fit a weak learner using the weighted training examples.

### Step 3: Calculate weighted error

Let (`\epsilon`{=tex}) represent the weighted error of the stump.

### Step 4: Calculate the stump's weight

The lecture gives:

\[
`\alpha`{=tex}=`\ln`{=tex}`\left`{=tex}(`\frac{1-\epsilon}{\epsilon}`{=tex}`\right`{=tex})
\]

A more accurate stump gets a larger vote in the final ensemble.

### Step 5: Increase the weights of misclassified examples

For a misclassified example, the lecture gives the update:

\[ w_i `\leftarrow `{=tex}w_i e\^`\alpha`{=tex} \]

The weights are then normalized.

The next stump pays more attention to the difficult examples.

## 10.3 Understanding the alpha values

If:

\[ `\epsilon=0.1`{=tex} \]

Then:

\[ `\alpha`{=tex}=`\ln`{=tex}(9)`\approx2.20`{=tex} \]

If:

\[ `\epsilon=0.4`{=tex} \]

Then:

\[ `\alpha`{=tex}=`\ln`{=tex}(1.5)`\approx0.41`{=tex} \]

If:

\[ `\epsilon=0.5`{=tex} \]

Then:

\[ `\alpha=0`{=tex} \]

A learner with 50% error receives no effective vote under this formula.

## 10.4 AdaBoost code

``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# A single decision stump
stump = DecisionTreeClassifier(
    max_depth=1,
    random_state=0
)

stump.fit(X_tr, y_tr)
print("One stump:", stump.score(X_te, y_te))

# AdaBoost with 200 weak learners
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    random_state=0
)

ada.fit(X_tr, y_tr)
print("AdaBoost:", ada.score(X_te, y_te))

# Inspect the first five learner errors and weights
print("Errors:", ada.estimator_errors_[:5].round(3))
print("Weights:", ada.estimator_weights_[:5].round(3))
```

### Results reported in the lecture

  Model         Accuracy
  ----------- ----------
  One stump        0.923
  AdaBoost         0.972

The ensemble combines many simple learners to produce a stronger model.

------------------------------------------------------------------------

# 11. Gradient Boosting

## 11.1 Main idea

Gradient Boosting trains a sequence of models. Each new model is fitted
to the remaining errors, or more generally to the negative gradient of
the loss function.

For regression with squared error, the residual is:

\[ r_i=y_i-F(x_i) \]

Where:

-   (y_i): actual target.
-   (F(x_i)): current prediction.
-   (r_i): residual.

For squared-error loss, this residual is exactly the negative gradient.

For other loss functions, such as classification log-loss, the model
fits the negative gradient rather than simply (y-`\hat `{=tex}y).

## 11.2 Step-by-step example

Suppose the actual target is 10 and the current prediction is 8.

The residual is:

\[ r=10-8=2 \]

Suppose the next tree predicts 2 and the learning rate is 0.1.

The updated prediction becomes:

\[ F\_{`\text{new}`{=tex}}=8+(0.1`\times2`{=tex})=8.2 \]

The new prediction is closer to 10.

## 11.3 Gradient Boosting update equation

The lecture gives:

\[ F_m(x)=F\_{m-1}(x)+`\nu `{=tex}h_m(x) \]

Where:

-   (F_m(x)): ensemble after adding tree (m).
-   (F\_{m-1}(x)): previous ensemble.
-   (h_m(x)): new tree fitted to the remaining errors/negative
    gradients.
-   (`\nu`{=tex}): learning rate, also called shrinkage.

## 11.4 Gradient Boosting workflow

``` mermaid
flowchart TD
    A[Start with a constant prediction] --> B[Calculate residuals / negative gradients]
    B --> C[Fit a small tree to them]
    C --> D[Multiply tree output by learning rate]
    D --> E[Add the update to the current model]
    E --> F{More boosting rounds?}
    F -- Yes --> B
    F -- No --> G[Final boosted model]
```

## 11.5 Gradient Boosting from scratch

This example follows the lecture's small regression demonstration.

``` python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

X = np.arange(1, 7).reshape(-1, 1)

y = np.array([
    5.0, 6.0, 7.5,
    12.0, 13.0, 14.5
])

# Start with the mean target value
pred = np.full(6, y.mean())

learning_rate = 0.5

for step in range(1, 5):
    # What is still wrong?
    residuals = y - pred

    # Fit a shallow tree to the residuals
    tree = DecisionTreeRegressor(
        max_depth=1
    )
    tree.fit(X, residuals)

    # Add a fraction of the new tree's prediction
    pred += learning_rate * tree.predict(X)

    mse = np.mean((y - pred) ** 2)

    print(
        f"round {step}: "
        f"residuals={residuals.round(1)}, "
        f"MSE={mse:.2f}"
    )
```

### Results reported in the lecture

    Round    MSE
  ------- ------
        1   4.12
        2   1.82
        3   0.93
        4   0.43

The MSE decreases over the rounds in this example.

------------------------------------------------------------------------

# 12. Learning Rate and Number of Trees

## 12.1 What is the learning rate?

The learning rate (`\nu`{=tex}) controls how much each new tree
contributes to the current ensemble.

\[ F_m(x)=F\_{m-1}(x)+`\nu `{=tex}h_m(x) \]

-   A larger learning rate makes each update larger.
-   A smaller learning rate makes each update smaller.

## 12.2 Learning-rate trade-off

A smaller learning rate usually requires more trees, but can improve
generalization when tuned appropriately.

The lecture lists these typical ranges:

  Parameter         Typical values in the lecture
  ----------------- -------------------------------
  `learning_rate`   0.01 to 0.3
  `n_estimators`    100 to 5000
  `max_depth`       2 to 6
  `subsample`       0.5 to 1.0

When `subsample < 1`, each tree is trained using a random part of the
data. This is called **stochastic gradient boosting**.

## 12.3 Can Boosting overfit?

Yes. Adding more and more trees can eventually fit noise in the training
data.

This is why validation monitoring and early stopping are useful.

## 12.4 Compare learning rates with staged predictions

``` python
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

for learning_rate in [1.0, 0.1, 0.01]:
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=learning_rate,
        max_depth=2,
        random_state=0
    )

    gb.fit(X_tr, y_tr)

    # Predictions after each added tree
    losses = [
        log_loss(y_te, probabilities)
        for probabilities in gb.staged_predict_proba(X_te)
    ]

    plt.plot(
        losses,
        label=f"learning_rate = {learning_rate}"
    )

plt.xlabel("Number of boosting stages")
plt.ylabel("Test log-loss (lower is better)")
plt.legend()
plt.show()
```

`staged_predict_proba` returns probabilities after each boosting stage,
allowing us to monitor how the loss changes as trees are added.

------------------------------------------------------------------------

# 13. Bagging vs. Boosting

  -----------------------------------------------------------------------
  Property                Bagging / Random Forest Boosting
  ----------------------- ----------------------- -----------------------
  Trees                   Often deep              Usually shallow

  Training                Independent and         Sequential
                          parallelizable          

  Main target             Reduce variance         Reduce bias (and
                                                  sometimes variance)

  Relationship between    Trained independently   Each stage improves the
  learners                                        current ensemble

  More trees              Usually stabilizes the  Can overfit if training
                          result                  continues too long

  Tuning                  Often works well with   Usually needs more
                          defaults                tuning
  -----------------------------------------------------------------------

**Practical starting point from the lecture:** start with a Random
Forest. If you need additional performance, try Gradient Boosting.

------------------------------------------------------------------------

# 14. Boosting at Scale: XGBoost and LightGBM

## 14.1 XGBoost

**XGBoost** stands for **eXtreme Gradient Boosting**.

The lecture identifies several important features.

### A. Regularized objective

XGBoost adds penalties to the loss to control model complexity:

\[ Obj = `\sum`{=tex}\_i l(y_i,`\hat `{=tex}y_i) + `\sum`{=tex}\_k
`\left`{=tex}( `\gamma `{=tex}T_k+
`\frac{1}{2}`{=tex}`\lambda`{=tex}\|w_k\|\^2 `\right`{=tex}) \]

Where:

-   (l): loss function, such as log-loss.
-   (T_k): number of leaves in tree (k).
-   (w_k): leaf values.
-   (`\gamma`{=tex}): penalty associated with the number of leaves.
-   (`\lambda`{=tex}): regularization parameter for leaf values.

The objective balances fitting the data with controlling complexity.

### B. Second-order optimization

XGBoost uses both:

-   The gradient (first derivative).
-   The Hessian (second derivative).

The lecture gives this leaf-value expression:

\[ w=-`\frac{G}{H+\lambda}`{=tex} \]

Where:

-   (G): sum of gradients.
-   (H): sum of Hessians.
-   (`\lambda`{=tex}): regularization parameter.

This is part of the second-order optimization approach often described
as Newton boosting.

### C. Speed and scalability

The lecture highlights:

-   Parallel split finding.
-   Histogram mode (`tree_method="hist"`).
-   GPU support.
-   Built-in missing-value handling.

### D. Common use

XGBoost became well known in tabular-data machine learning competitions,
including Kaggle competitions.

## 14.2 LightGBM

**LightGBM** stands for **Light Gradient Boosting Machine**. The lecture
identifies Microsoft (2017) as its origin.

### A. Histogram-based splitting

LightGBM bins feature values into discrete buckets. The lecture
describes up to 255 buckets per feature.

Instead of evaluating every individual feature value as a possible
split, the algorithm scans the bins. This can make split finding faster.

### B. Leaf-wise tree growth

LightGBM grows the leaf that gives the largest loss reduction rather
than growing every level uniformly.

``` mermaid
flowchart TD
    A[Find candidate leaves] --> B[Calculate loss reduction]
    B --> C[Choose leaf with largest gain]
    C --> D[Split selected leaf]
    D --> A
```

Leaf-wise growth can create deep trees quickly, so control:

-   `num_leaves`
-   `min_child_samples`

### C. GOSS

**Gradient-based One-Side Sampling** keeps examples with large gradients
and samples from examples with smaller gradients.

### D. EFB

**Exclusive Feature Bundling** bundles sparse features that are rarely
non-zero at the same time. This can reduce the effective number of
features.

## 14.3 XGBoost vs. LightGBM: lecture-level summary

  -----------------------------------------------------------------------
  Feature                 XGBoost                 LightGBM
  ----------------------- ----------------------- -----------------------
  Regularization          Explicit regularized    Supports regularization
                          objective               settings

  Tree growth             Commonly level-wise in  Leaf-wise
                          standard configurations 

  Histogram approach      Supported               Core efficiency
                                                  technique

  Special techniques      Second-order            GOSS and EFB
  highlighted             optimization            

  Important complexity    Depth and               `num_leaves`,
  control                 regularization          `min_child_samples`
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 15. Histogram Gradient Boosting

Scikit-learn provides `HistGradientBoostingClassifier`, a
histogram-based gradient boosting implementation.

The lecture compares:

-   `GradientBoostingClassifier`
-   `HistGradientBoostingClassifier`

using a synthetic dataset with 50,000 rows and 20 features.

## 15.1 Example code

``` python
from time import perf_counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

X, y = make_classification(
    n_samples=50_000,
    n_features=20,
    n_informative=10,
    random_state=0
)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    random_state=0
)

for Model in [
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
]:
    start = perf_counter()

    model = Model(
        random_state=0
    ).fit(X_tr, y_tr)

    seconds = perf_counter() - start
    accuracy = model.score(X_te, y_te)

    print(
        f"{Model.__name__:30} "
        f"{seconds:4.1f}s "
        f"accuracy {accuracy:.3f}"
    )
```

### Results reported in the lecture

  Model                              Training time   Test accuracy
  -------------------------------- --------------- ---------------
  GradientBoostingClassifier                23.7 s           0.942
  HistGradientBoostingClassifier             0.6 s           0.977

These are the results from the lecture's specific experiment, not a
guarantee for every dataset or machine.

------------------------------------------------------------------------

# 16. Early Stopping

## 16.1 What is Early Stopping?

Early stopping monitors performance on validation data and stops
training when the validation score has not improved for a specified
number of iterations.

It helps to:

-   Reduce overfitting.
-   Save training time.
-   Avoid choosing an unnecessarily large number of trees manually.

## 16.2 Histogram Gradient Boosting example

``` python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=0
)

model.fit(X_tr, y_tr)
```

Parameter meanings:

-   `early_stopping=True`: enable early stopping.
-   `validation_fraction=0.1`: reserve 10% of the training data for
    validation.
-   `n_iter_no_change=10`: stop if the monitored score does not improve
    for the specified number of iterations.

The lecture notes that histogram boosting enables early stopping by
default above 10,000 rows, unless its behavior is changed through the
settings.

## 16.3 Classic Gradient Boosting

``` python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_iter_no_change=10,
    random_state=0
)

model.fit(X_tr, y_tr)
```

## 16.4 XGBoost and LightGBM

Both libraries also support early stopping using validation data, with
library-specific APIs:

-   XGBoost: `early_stopping_rounds` (depending on the API/version).
-   LightGBM: an early-stopping callback.

------------------------------------------------------------------------

# 17. Important Equations

  --------------------------------------------------------------------------------------------------------------------
  Topic                   Equation                                                             Meaning
  ----------------------- -------------------------------------------------------------------- -----------------------
  Bootstrap               (1-(1-`\frac{1}{n}`{=tex})\^n)                                       Probability a row
                                                                                               appears at least once

  Large-sample Bootstrap  (1-`\frac{1}{e}`{=tex}`\approx0.632`{=tex})                          Expected distinct-row
                                                                                               fraction

  Bagging                 (f\_{bag}(x)=`\frac{1}{B}`{=tex}`\sum`{=tex}\_{b=1}\^{B}f_b(x))      Average model
                                                                                               prediction

  Ensemble variance       (`\rho`{=tex}`\sigma`{=tex}\^2+`\frac{(1-\rho)\sigma^2}{B}`{=tex})   Variance of an average
                                                                                               of correlated models

  Random Forest features  (`\sqrt{p}`{=tex})                                                   Typical classification
                                                                                               feature subset size

  AdaBoost weight         (`\alpha`{=tex}=`\ln`{=tex}(`\frac{1-\epsilon}{\epsilon}`{=tex}))    Weight of a weak
                                                                                               learner

  Residual                (r_i=y_i-F(x_i))                                                     Remaining error for
                                                                                               squared-error
                                                                                               regression

  Gradient Boosting       (F_m=F\_{m-1}+`\nu `{=tex}h_m)                                       Add a scaled new
                                                                                               learner

  XGBoost leaf value      (w=-`\frac{G}{H+\lambda}`{=tex})                                     Leaf value using
                                                                                               gradient and Hessian
                                                                                               sums
  --------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 18. Quick Reference

## 18.1 Key terms

  Term                 Definition
  -------------------- --------------------------------------------------------
  Ensemble Learning    Combining multiple models
  Hard Voting          Majority vote over predicted labels
  Soft Voting          Averaging predicted class probabilities
  Bootstrap            Sampling with replacement
  Bagging              Bootstrap Aggregating
  OOB                  Out-of-Bag samples not used to train a particular tree
  Bias                 Error due to an overly simple model
  Variance             Sensitivity to changes in training data
  Weak learner         A simple model used as a building block
  Decision stump       A tree with one split
  Residual             Actual target minus current prediction
  Learning rate        Controls the size of each boosting update
  Sequential           Models are trained one after another
  Parallel             Independent models can be trained at the same time
  Early stopping       Stop when validation performance stops improving
  Feature importance   Estimate of how much a model relies on features

## 18.2 Which method should I try?

  Situation                                       Method to consider
  ----------------------------------------------- ----------------------
  Combine different model types                   Voting
  Reduce variance of unstable models              Bagging
  Need a strong tree-based baseline               Random Forest
  Focus on previously misclassified examples      AdaBoost
  Fit remaining errors stage by stage             Gradient Boosting
  Need scalable gradient boosting                 XGBoost or LightGBM
  Need histogram-based boosting in scikit-learn   HistGradientBoosting

## 18.3 Core takeaways

1.  Ensemble Learning combines predictions from multiple models.
2.  Useful ensembles need models that are better than random and do not
    all make identical errors.
3.  Bagging trains models independently on bootstrap samples and mainly
    reduces variance.
4.  Random Forest adds random feature selection to reduce correlation
    between trees.
5.  OOB evaluation uses training rows excluded from a tree's bootstrap
    sample.
6.  Boosting trains weak learners sequentially.
7.  AdaBoost increases the weights of misclassified examples.
8.  Gradient Boosting fits residuals or negative gradients.
9.  Learning rate controls the contribution of each new tree.
10. XGBoost and LightGBM provide scalable gradient boosting approaches.
11. Early stopping can reduce unnecessary training and help prevent
    overfitting.

------------------------------------------------------------------------

# 19. Practice Questions

Try to answer these questions before checking the lecture notes again.

1.  What is Ensemble Learning, and why can combining models help?
2.  What is the difference between Bias and Variance?
3.  Which problem does Bagging mainly reduce?
4.  Which problem does Boosting mainly reduce?
5.  What is the difference between Hard Voting and Soft Voting?
6.  What does sampling with replacement mean?
7.  Why does a large bootstrap sample contain approximately 63.2%
    distinct rows?
8.  What does OOB stand for, and how is an OOB prediction produced?
9.  What additional randomization does Random Forest introduce beyond
    Bagging?
10. Why can reducing correlation between trees improve a forest?
11. What is a Decision Stump?
12. How does AdaBoost change the weights of training examples?
13. What is a residual in squared-error regression?
14. What does the learning rate control in Gradient Boosting?
15. What is the purpose of Early Stopping?
16. What is the difference between XGBoost and LightGBM's tree-growth
    strategies as described in the lecture?

------------------------------------------------------------------------

## Suggested environment

Install the core libraries:

``` bash
pip install numpy pandas matplotlib scikit-learn
```

For the optional boosting libraries:

``` bash
pip install xgboost lightgbm
```

> **Version note:** scikit-learn APIs can differ between versions. For
> example, newer versions use the `estimator=` parameter in
> `BaggingClassifier` and `AdaBoostClassifier`; older versions may use
> `base_estimator=`. Check your installed version's documentation if an
> argument is rejected.

------------------------------------------------------------------------

**Lecture source:** *Ensemble Learning --- TriNode Team, 37 slides.*\
**End of README.**
