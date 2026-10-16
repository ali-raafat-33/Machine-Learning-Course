# Logistic Regression — One-Page Cheat Sheet

**Goal:** predict P(y = 1 | x) for a binary target, then classify.

## Model
```
z = w·x + b            # linear score (any real number)
p = σ(z) = 1/(1+e⁻ᶻ)   # probability in (0, 1)
ŷ = 1 if p ≥ 0.5 else 0
```
- σ(0)=0.5 · σ(−z)=1−σ(z) · σ′=σ(1−σ)
- Linear in log-odds: `log(p/(1−p)) = w·x + b`
- Weight → odds ratio: **e^w** per +1 in the feature (w>0 ↑ class 1, w<0 ↓)
- Boundary: `w·x + b = 0` (line / plane → linear classifier)

## Training
```
loss  = −mean[ y·log p + (1−y)·log(1−p) ]    # log-loss, convex
∂J/∂w = mean[(p−y)·x]    ∂J/∂b = mean[p−y]
w ← w − η·∂J/∂w          b ← b − η·∂J/∂b
```
- No closed form → gradient descent / L-BFGS
- Predicting 0.5 everywhere → loss = ln 2 ≈ 0.693
- Regularization: L2 (default) / L1; **small C = strong penalty**

## Multiclass
Softmax `p_k = e^{z_k} / Σ_j e^{z_j}`; loss = −log p(true class). (or One-vs-Rest)

## Metrics
| Metric | Formula |
|---|---|
| Accuracy | (TP+TN)/all |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| F1 | 2PR/(P+R) |
| ROC-AUC | area under TPR-vs-FPR; 0.5 random, 1 perfect |
| Log-loss | quality of probabilities |

Lower threshold → more recall · Higher threshold → more precision.

## Workflow
1. **Prepare**: clean → encode → split → scale (fit on train only)
2. **Train**: `make_pipeline(StandardScaler(), LogisticRegression())`; tune C by CV
3. **Evaluate**: confusion matrix, P/R/F1, ROC-AUC, threshold
4. **Interpret**: signs, odds ratios (on standardised features), calibration

## Assumptions
Binary target · independent rows · log-odds linear in features · low multicollinearity · enough data · few extreme outliers.

## Pitfalls
Accuracy on imbalanced data · leakage · unscaled features · perfect separation · multicollinearity · always using 0.5 · using MSE.

## scikit-learn snippet
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)
```
