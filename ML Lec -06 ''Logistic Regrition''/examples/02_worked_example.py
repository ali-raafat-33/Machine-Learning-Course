"""Worked example: 12 students, hours studied -> pass/fail.

Fits the model twice (from scratch with gradient descent, and with scikit-learn)
and prints the numbers used in the README.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "hours_studied.csv")
x, y = df["hours_studied"].to_numpy(), df["passed"].to_numpy()
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# --- one gradient-descent step by hand (w = b = 0 -> every p = 0.5) -------------
p0 = np.full_like(x, 0.5)
gw, gb = np.mean((p0 - y) * x), np.mean(p0 - y)
print(f"Step 1 gradients: dJ/dw = {gw:.4f}, dJ/db = {gb:.4f}")
print(f"After one step (lr=0.1): w = {-0.1 * gw:.4f}, b = {-0.1 * gb:.4f}\n")

# --- full training from scratch --------------------------------------------------
w, b, lr = 0.0, 0.0, 0.1
for epoch in range(100_001):
    p = sigmoid(w * x + b)
    w -= lr * np.mean((p - y) * x)
    b -= lr * np.mean(p - y)
    if epoch in (0, 1_000, 10_000, 100_000):
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        print(f"epoch {epoch:6}: w={w:.3f} b={b:.3f} loss={loss:.4f}")
print(f"\nfrom scratch : w={w:.3f}, b={b:.3f}, boundary={-b / w:.2f} h, odds ratio={np.exp(w):.2f}")

# --- scikit-learn (C very large ~ no regularisation) -----------------------------
clf = LogisticRegression(C=1e10).fit(x[:, None], y)
print(f"scikit-learn : w={clf.coef_[0][0]:.3f}, b={clf.intercept_[0]:.3f}")

# --- predictions & metrics -------------------------------------------------------
proba = clf.predict_proba(x[:, None])[:, 1]
pred = (proba >= 0.5).astype(int)
print("\nhours  P(pass)  pred  actual")
for h, pr, pd_, yy in zip(x, proba, pred, y):
    print(f"{h:5.1f}  {pr:7.3f}  {pd_:4}  {yy:6}")
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}")
print(f"accuracy={(tp + tn) / len(y):.2f} precision={tp / (tp + fp):.2f} recall={tp / (tp + fn):.2f}")
print(f"log-loss={log_loss(y, proba):.3f}")

# --- a new student ---------------------------------------------------------------
h = 4.0
z = clf.coef_[0][0] * h + clf.intercept_[0]
print(f"\nStudent with {h} h: z={z:.3f} -> p={sigmoid(z):.3f} -> class {int(sigmoid(z) >= .5)}")
