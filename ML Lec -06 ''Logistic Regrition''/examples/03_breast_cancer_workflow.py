"""End-to-end workflow on scikit-learn's breast-cancer dataset (569 samples, 30 features)."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Data preparation ------------------------------------------------------------
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=0)

# 2. Training (scaler lives inside the pipeline -> no leakage during CV) ----------
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="roc_auc")
grid.fit(X_tr, y_tr)
model = grid.best_estimator_
print("best C:", grid.best_params_["logisticregression__C"], "| CV AUC:", round(grid.best_score_, 4))

# 3. Evaluation on the untouched test set ----------------------------------------
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]
print("\nConfusion matrix (rows = actual, cols = predicted; 0 = malignant, 1 = benign)")
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=data.target_names))
print("test ROC-AUC:", round(roc_auc_score(y_te, proba), 4))

# 4. Interpretation: weights are on *standardised* features -----------------------
coefs = model.named_steps["logisticregression"].coef_[0]
top = np.argsort(np.abs(coefs))[::-1][:5]
print("\nTop 5 features (odds ratio per +1 std-dev; >1 pushes toward 'benign')")
for i in top:
    print(f"  {data.feature_names[i]:<25} w={coefs[i]:+.2f}  OR={np.exp(coefs[i]):.2f}")
