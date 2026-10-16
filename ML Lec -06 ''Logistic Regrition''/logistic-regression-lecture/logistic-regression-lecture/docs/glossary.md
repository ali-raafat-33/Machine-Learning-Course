# Glossary

| Term | Meaning |
|---|---|
| **Binary classification** | Predicting one of two classes (0/1). |
| **Sigmoid / logistic function** | σ(z) = 1 / (1 + e⁻ᶻ); maps any real number to (0, 1). |
| **Logit / log-odds** | log(p / (1 − p)); the quantity that is linear in the features. |
| **Odds** | p / (1 − p); how many times more likely success is than failure. |
| **Odds ratio** | e^w; factor by which the odds change for +1 in a feature. |
| **Weights (w), bias (b)** | Parameters learned in training; `coef_` and `intercept_` in scikit-learn. |
| **Decision boundary** | Set of points where p = 0.5, i.e. w·x + b = 0. |
| **Threshold** | Probability cut-off for predicting class 1 (default 0.5). |
| **Log-loss / cross-entropy** | −[y·log p + (1 − y)·log(1 − p)] averaged over samples; the training cost. |
| **Maximum likelihood** | Choosing parameters that make the observed labels most probable; equivalent to minimising log-loss. |
| **Convex** | Having a single (global) minimum, so gradient descent cannot get stuck in local minima. |
| **Gradient descent** | Iteratively moving parameters against the gradient of the loss. |
| **Learning rate (η)** | Step size of each gradient-descent update. |
| **Regularization** | Penalty on weight size to reduce overfitting (L1 / L2). |
| **C** | Inverse regularization strength in scikit-learn (small C = stronger penalty). |
| **Softmax** | Generalises the sigmoid to k classes; probabilities sum to 1. |
| **One-vs-Rest (OvR)** | Multiclass strategy: one binary classifier per class. |
| **Confusion matrix** | Table of TP, FP, FN, TN counts. |
| **Precision** | TP / (TP + FP): when it says positive, how often is it right? |
| **Recall (sensitivity)** | TP / (TP + FN): of all real positives, how many were found? |
| **F1 score** | 2PR / (P + R); harmonic mean of precision and recall. |
| **ROC curve / AUC** | True-positive rate vs false-positive rate over all thresholds; AUC 1.0 = perfect, 0.5 = random. |
| **Class imbalance** | One class is much rarer than the other; accuracy becomes misleading. |
| **Multicollinearity** | Strong correlation between features; destabilises weights. |
| **Perfect separation** | A line splits the classes exactly; weights diverge without regularization. |
| **Data leakage** | Information from the test set influencing training (e.g. scaling before the split). |
