"""Generate every figure used in the README -> ../images/*.png

Run:  python examples/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN, AMBER = "#2b6cb0", "#c53030", "#718096", "#2f855a", "#d69e2e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})

sigmoid = lambda z: 1 / (1 + np.exp(-z))
df = pd.read_csv(ROOT / "data" / "hours_studied.csv")
X, y = df[["hours_studied"]].values, df["passed"].values
clf = LogisticRegression(C=1e10).fit(X, y)          # ~ unregularised
w, b = clf.coef_[0][0], clf.intercept_[0]


def save(name):
    plt.tight_layout()
    plt.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", name)


# 1 ── Linear vs logistic -------------------------------------------------------
def fig_linear_vs_logistic():
    xs = np.linspace(-1, 8, 200)
    lin = LinearRegression().fit(X, y)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(X, y, c=np.where(y == 1, RED, BLUE), s=55, zorder=3, edgecolor="k", lw=.5)
    ax.plot(xs, lin.predict(xs[:, None]), c=BLUE, lw=2.5, label="Linear regression")
    ax.plot(xs, sigmoid(w * xs + b), c=RED, lw=2.5, label="Logistic regression")
    ax.axhspan(-0.6, 0, color=GREY, alpha=.12); ax.axhspan(1, 1.6, color=GREY, alpha=.12)
    ax.text(7.9, -.3, "impossible as a probability", ha="right", color=GREY, fontsize=9)
    ax.text(7.9, 1.3, "impossible as a probability", ha="right", color=GREY, fontsize=9)
    ax.set(xlim=(-1, 8), ylim=(-.6, 1.6), xlabel="hours studied", ylabel="pass (1) / fail (0)",
           title="A straight line escapes [0, 1]; the sigmoid never does")
    ax.legend(loc="center right", frameon=False)
    save("01_linear_vs_logistic.png")


# 2 ── Sigmoid ----------------------------------------------------------------
def fig_sigmoid():
    z = np.linspace(-8, 8, 300)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(z, sigmoid(z), c=RED, lw=3)
    ax.axhline(.5, ls="--", c=GREY, lw=1); ax.axvline(0, ls=":", c=GREY, lw=1)
    for zz in (-4, 0, 2, 4):
        ax.scatter(zz, sigmoid(zz), c=BLUE, zorder=3)
        ax.annotate(f"σ({zz}) = {sigmoid(zz):.3f}", (zz, sigmoid(zz)),
                    textcoords="offset points", xytext=(8, -14 if zz >= 0 else 8), fontsize=9)
    ax.set(xlabel="z = w·x + b  (linear score)", ylabel="p = σ(z)",
           title="The sigmoid: any real number → a probability", ylim=(-.05, 1.05))
    ax.text(-7.8, .56, "threshold 0.5", color=GREY, fontsize=9)
    save("02_sigmoid.png")


# 3 ── Concept diagram: from features to a class -------------------------------
def box(ax, xy, w_, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w_, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=color, ec="none", alpha=.92))
    ax.text(xy[0] + w_ / 2, xy[1] + h / 2, text, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")


def arrow(ax, a, b_):
    ax.add_patch(FancyArrowPatch(a, b_, arrowstyle="-|>", mutation_scale=16, color="#2d3748", lw=1.6))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(.25, 2.4)
    items = [("Features\nx₁ … xₙ", GREY), ("Linear score\nz = w·x + b", BLUE),
             ("Sigmoid\np = 1/(1+e⁻ᶻ)", RED), ("Threshold\np ≥ 0.5 ?", AMBER), ("Class\n0 or 1", GREEN)]
    xs = [0.1, 2.3, 4.5, 6.7, 8.9]
    for (t, c), x0 in zip(items, xs):
        box(ax, (x0, 1.0), 1.9, 1.1, t, c)
    for x0 in xs[:-1]:
        arrow(ax, (x0 + 1.95, 1.55), (x0 + 2.25, 1.55))
    notes = ["any real numbers", "any real number\n(−∞ … +∞)", "probability\n(0 … 1)", "decision rule\n(tunable)", "prediction"]
    for n, x0 in zip(notes, xs):
        ax.text(x0 + .95, .55, n, ha="center", va="center", fontsize=10, color="#4a5568")
    ax.set_title("Logistic regression in one picture", loc="left")
    save("03_pipeline.png")


# 4 ── Decision boundary ------------------------------------------------------
def fig_boundary():
    rng = np.random.default_rng(1)
    X0 = rng.normal([40, 45], 11, (60, 2)); X1 = rng.normal([70, 72], 11, (60, 2))
    Xs = np.vstack([X0, X1]); ys = np.r_[np.zeros(60), np.ones(60)]
    m = LogisticRegression().fit(Xs, ys); w_, b_ = m.coef_[0], m.intercept_[0]
    a, c = np.meshgrid(np.linspace(10, 105, 300), np.linspace(10, 110, 300))
    P = m.predict_proba(np.c_[a.ravel(), c.ravel()])[:, 1].reshape(a.shape)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    cf = ax.contourf(a, c, P, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=.55)
    ax.contour(a, c, P, levels=[.5], colors="k", linewidths=2.2, linestyles="--")
    ax.scatter(*Xs.T, c=ys, cmap="coolwarm", edgecolor="k", lw=.5, s=32)
    plt.colorbar(cf, ax=ax, label="P(y = 1)")
    ax.set(xlabel="feature x₁ (exam 1)", ylabel="feature x₂ (exam 2)",
           title="Decision boundary: w₁x₁ + w₂x₂ + b = 0")
    ax.text(13, 13, "far from the line → confident", fontsize=9)
    ax.annotate("p = 0.5", (60, 56), xytext=(78, 22), arrowprops=dict(arrowstyle="->"), fontsize=10)
    save("04_decision_boundary.png")


# 5 ── Log-loss vs MSE ----------------------------------------------------------
def fig_logloss():
    p = np.linspace(.005, .995, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(p, -np.log(p), c=RED, lw=2.5, label="true y = 1:  −log(p)")
    ax.plot(p, -np.log(1 - p), c=BLUE, lw=2.5, label="true y = 0:  −log(1−p)")
    ax.set(ylim=(0, 5), xlabel="predicted probability p", ylabel="loss", title="Log-loss per sample")
    ax.legend(frameon=False, loc="upper center")
    ax = axes[1]
    ax.plot(p, -np.log(p), c=RED, lw=2.5, label="log-loss")
    ax.plot(p, (1 - p) ** 2, c=GREY, lw=2.5, ls="--", label="squared error")
    ax.set(ylim=(0, 5), xlabel="predicted probability p  (true y = 1)", title="Confident mistakes: log-loss explodes")
    ax.legend(frameon=False, loc="upper right")
    save("05_log_loss.png")


# 6 ── Workflow ---------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.4)
    steps = [("1  Prepare data", ["clean & impute", "encode categories", "scale features", "split train / test"], GREY),
             ("2  Train", ["pick C / penalty", "minimise log-loss", "gradient descent"], BLUE),
             ("3  Evaluate", ["confusion matrix", "precision / recall / F1", "ROC-AUC, log-loss", "tune threshold"], AMBER),
             ("4  Interpret", ["read weights", "odds ratio  eᵂ", "check calibration"], GREEN)]
    xs = [0.1, 2.85, 5.6, 8.35]
    for (t, bl, c), x0 in zip(steps, xs):
        box(ax, (x0, 2.2), 2.5, .8, t, c)
        for i, line in enumerate(bl):
            ax.text(x0 + .15, 1.85 - i * .38, "• " + line, fontsize=9.5, color="#2d3748")
    for x0 in xs[:-1]:
        arrow(ax, (x0 + 2.55, 2.6), (x0 + 2.8, 2.6))
    ax.set_title("Typical workflow", loc="left")
    save("06_workflow.png")


# 7 ── Worked example ---------------------------------------------------------
def fig_worked():
    xs = np.linspace(0, 6.5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(X, y, c=np.where(y == 1, RED, BLUE), s=70, edgecolor="k", lw=.6, zorder=3)
    ax.plot(xs, sigmoid(w * xs + b), c=RED, lw=2.8, label=f"p = σ({w:.2f}·hours {b:+.2f})")
    xb = -b / w
    ax.axhline(.5, ls="--", c=GREY, lw=1); ax.axvline(xb, ls=":", c="k", lw=1.2)
    ax.annotate(f"boundary ≈ {xb:.2f} h", (xb, .5), xytext=(xb + .3, .15),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.set(xlabel="hours studied", ylabel="P(pass)", ylim=(-.08, 1.08),
           title="Worked example: 12 students, hours studied → pass")
    ax.legend(frameon=False, loc="upper left")
    save("07_worked_example.png")


# 8 ── Confusion matrix + metrics --------------------------------------------
def fig_confusion():
    pred = (clf.predict_proba(X)[:, 1] >= .5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]; ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    cells = [(0, 1, "TP", tp, GREEN), (1, 1, "FN", fn, RED), (0, 0, "FP", fp, RED), (1, 0, "TN", tn, GREEN)]
    for cx, cy, name, val, col in cells:
        ax.add_patch(plt.Rectangle((cx, cy), 1, 1, fc=col, alpha=.25, ec="k"))
        ax.text(cx + .5, cy + .58, name, ha="center", fontsize=15, fontweight="bold")
        ax.text(cx + .5, cy + .3, str(val), ha="center", fontsize=13)
    ax.set_xticks([.5, 1.5], ["Predicted 1", "Predicted 0"]); ax.set_yticks([.5, 1.5], ["Actual 0", "Actual 1"])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Confusion matrix (worked example)")
    ax = axes[1]; ax.axis("off")
    acc = (tp + tn) / len(y); prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec)
    rows = [("Accuracy", "(TP+TN) / all", f"{acc:.2f}"), ("Precision", "TP / (TP+FP)", f"{prec:.2f}"),
            ("Recall", "TP / (TP+FN)", f"{rec:.2f}"), ("F1", "2PR / (P+R)", f"{f1:.2f}")]
    for i, (n, f, v) in enumerate(rows):
        yy = .85 - i * .22
        ax.text(0, yy, n, fontweight="bold", fontsize=12, transform=ax.transAxes)
        ax.text(.33, yy, f, fontsize=11, color="#4a5568", transform=ax.transAxes)
        ax.text(.9, yy, v, fontsize=12, ha="right", transform=ax.transAxes)
    ax.set_title("Metrics at threshold 0.5")
    save("08_confusion_matrix.png")


if __name__ == "__main__":
    for f in (fig_linear_vs_logistic, fig_sigmoid, fig_pipeline, fig_boundary,
              fig_logloss, fig_workflow, fig_worked, fig_confusion):
        f()
