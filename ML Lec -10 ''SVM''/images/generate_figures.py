import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles

import os
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
RED, BLUE, YEL, GREY = "#d6453d", "#2f6db5", "#ffe08a", "#555555"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid_for(ax_lim, n=300):
    a, b = np.meshgrid(np.linspace(ax_lim[0], ax_lim[1], n),
                       np.linspace(ax_lim[2], ax_lim[3], n))
    return a, b, np.c_[a.ravel(), b.ravel()]


def scatter2(ax, X, y, s=45):
    ax.scatter(*X[y == 1].T, c=RED, s=s, edgecolor="white", lw=.6, label="class +1", zorder=3)
    ax.scatter(*X[y == -1].T, c=BLUE, s=s, edgecolor="white", lw=.6, label="class -1", zorder=3)


# ---------------------------------------------------------------- separable data
rng = np.random.RandomState(3)
Xp = rng.randn(18, 2) * .7 + [5.2, 4.8]
Xn = rng.randn(18, 2) * .7 + [2.2, 1.6]
X = np.vstack([Xp, Xn]); y = np.r_[np.ones(18), -np.ones(18)]
lim = (-0.2, 8, -0.5, 7.5)

# ---- 1. many lines vs max margin
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
xs = np.linspace(lim[0], lim[1], 10)
ax = axs[0]; scatter2(ax, X, y)
cx, cy = X.mean(0)
for slope, off, col in [(-0.9, 0.9, "#888"), (-0.25, -.25, "#888"), (-2.5, 4.5, "#888")]:
    ax.plot(xs, slope * (xs - cx) + cy + off, c=col, lw=1.6)
ax.set(xlim=lim[:2], ylim=lim[2:], title="Many lines separate the data...\nwhich one is best?")
ax.legend(loc="upper right", fontsize=9)
ax = axs[1]; scatter2(ax, X, y)
m = SVC(kernel="linear", C=1000).fit(X, y)
a, b, G = grid_for(lim); Z = m.decision_function(G).reshape(a.shape)
ax.contourf(a, b, Z, levels=[-1, 1], colors=[YEL], alpha=.6)
ax.contour(a, b, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
ax.set(xlim=lim[:2], ylim=lim[2:], title="SVM picks the middle of the\nwidest empty 'street'")
save(fig, "01_many_lines_vs_max_margin.png")

# ---- 2. hyperplane w.x + b = 0
fig, ax = plt.subplots(figsize=(6.4, 5.2))
w = np.array([2., 1.]); b0 = -4.
xx = np.linspace(-1, 5, 50)
ax.plot(xx, 4 - 2 * xx, "k", lw=2.4)
ax.fill_between(xx, 4 - 2 * xx, 8, color=RED, alpha=.08)
ax.fill_between(xx, -3, 4 - 2 * xx, color=BLUE, alpha=.08)
ax.text(-.6, 6.6, "w·x + b > 0   →  class +1", color=RED, fontsize=11, weight="bold")
ax.text(2.1, -2.4, "w·x + b < 0   →  class -1", color=BLUE, fontsize=11, weight="bold")
pts = np.array([[1, 1], [3, 2], [2, 0], [0, 6]])
for p in pts:
    s = p @ w + b0
    col = RED if s > 0 else (BLUE if s < 0 else "k")
    ax.scatter(*p, c=col, s=90, zorder=4, edgecolor="white")
    ax.annotate(f"({p[0]},{p[1]})  score={s:+.0f}", p, xytext=(8, 6), textcoords="offset points", fontsize=9)
# w arrow from point on the line (1,2)
ax.annotate("", xy=(1 + .8, 2 + .4), xytext=(1, 2), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#2a9d4a"))
ax.text(1.95, 2.5, "w = (2, 1)\n(perpendicular to line)", color="#2a9d4a", fontsize=9.5)
# perpendicular distance for (3,2)
foot = np.array([3, 2]) - (3 * 2 + 2 - 4) / 5 * w
ax.plot([3, foot[0]], [2, foot[1]], "--", c="#2a9d4a", lw=1.6)
ax.text(2.55, 1.05, "distance\n= 4/√5 = 1.79", fontsize=9, color="#2a9d4a", ha="right")
ax.set(xlim=(-1, 5), ylim=(-3, 7.2), title="Hyperplane  2x₁ + x₂ − 4 = 0  (a line in 2-D)")
ax.set_aspect("equal")
save(fig, "02_hyperplane.png")

# ---- 3. margin + support vectors
fig, ax = plt.subplots(figsize=(6.6, 5.2))
scatter2(ax, X, y)
ax.contourf(a, b, Z, levels=[-1, 1], colors=[YEL], alpha=.6)
ax.contour(a, b, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1.3, 2.4, 1.3])
sv = m.support_vectors_
ax.scatter(*sv.T, s=260, fc="none", ec="k", lw=2, zorder=5, label="support vectors")
ax.text(0.3, 6.8, "w·x + b = −1", fontsize=9); ax.text(0.3, 6.1, "w·x + b = 0", fontsize=9, weight="bold")
ax.set(xlim=lim[:2], ylim=lim[2:], title="Margin = yellow street · Support vectors = circled points")
ax.legend(loc="lower right", fontsize=9)
save(fig, "03_margin_support_vectors.png")

# ---- 4. margin geometry 2/||w||  (uses the lecture's trained w, b)
wl = np.array([0.427, 0.56]); bl = -4.217
rng2 = np.random.RandomState(1)
Pp, Pn = [], []
while len(Pp) < 9 or len(Pn) < 9:
    p = rng2.uniform([0, 0], [9, 8])
    s = p @ wl + bl
    if s > 1.4 and len(Pp) < 9 and np.linalg.norm(p - [6.6, 5.2]) < 2.6: Pp.append(p)
    if s < -1.4 and len(Pn) < 9 and np.linalg.norm(p - [3.6, 2.0]) < 2.6: Pn.append(p)
Pp.append([5.96, 4.77]); Pn.append([4.24, 2.51])
Pp, Pn = np.array(Pp), np.array(Pn)
fig, ax = plt.subplots(figsize=(7, 5.6))
ax.scatter(*Pp.T, c=RED, s=50, edgecolor="white", zorder=3); ax.scatter(*Pn.T, c=BLUE, s=50, edgecolor="white", zorder=3)
x1 = np.linspace(-1, 10, 20)
for lvl, ls, lw in [(-1, "--", 1.3), (0, "-", 2.4), (1, "--", 1.3)]:
    ax.plot(x1, (lvl - bl - wl[0] * x1) / wl[1], "k", ls=ls, lw=lw)
ax.fill_between(x1, (-1 - bl - wl[0] * x1) / wl[1], (1 - bl - wl[0] * x1) / wl[1], color=YEL, alpha=.5)
ax.scatter([4.24, 5.96], [2.51, 4.77], s=260, fc="none", ec="k", lw=2, zorder=5)
ax.annotate("", xy=(5.96, 4.77), xytext=(4.24, 2.51), arrowprops=dict(arrowstyle="<->", lw=2.2, color="#8a2be2"))
ax.text(4.85, 4.0, "margin\n= 2/‖w‖\n= 2.84", color="#8a2be2", fontsize=11, weight="bold", ha="right", va="center",
        bbox=dict(fc="white", ec="none", alpha=.8, pad=2))
p0 = np.array([7.2, (-bl - wl[0] * 7.2) / wl[1]])
ax.annotate("", xy=p0 + wl / np.linalg.norm(wl) * 1.4, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#2a9d4a"))
ax.text(p0[0] + .75, p0[1] + .85, "w", color="#2a9d4a", fontsize=15, weight="bold")
ax.text(.6, 7.15, "w·x + b = +1", rotation=-37, fontsize=9)
ax.text(.6, 5.6, "w·x + b = 0", rotation=-37, fontsize=9, weight="bold")
ax.text(.6, 4.0, "w·x + b = −1", rotation=-37, fontsize=9)
ax.set(xlim=(0, 9), ylim=(0, 8), title="Margin width = 2 / ‖w‖   (smaller ‖w‖ → wider street)")
ax.set_aspect("equal")
save(fig, "04_margin_geometry.png")

# ---------------------------------------------------------------- overlapping data
Xo, yo = make_blobs(n_samples=70, centers=[[2.6, 2.4], [4.6, 4.4]], cluster_std=1.05, random_state=11)
yo = np.where(yo == 1, 1, -1)
limo = (-0.5, 8, -0.5, 7.5)

# ---- 5. soft margin and slack
mo = SVC(kernel="linear", C=1).fit(Xo, yo)
fo = mo.decision_function(Xo); xi = np.maximum(0, 1 - yo * fo)
ao, bo, Go = grid_for(limo); Zo = mo.decision_function(Go).reshape(ao.shape)
fig, ax = plt.subplots(figsize=(7, 5.6))
ax.contourf(ao, bo, Zo, levels=[-1, 1], colors=[YEL], alpha=.55)
ax.contour(ao, bo, Zo, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
ok = xi == 0
ax.scatter(*Xo[ok & (yo == 1)].T, c=RED, s=40, edgecolor="white", zorder=3)
ax.scatter(*Xo[ok & (yo == -1)].T, c=BLUE, s=40, edgecolor="white", zorder=3)
wo = mo.coef_[0]
for i in np.where(~ok)[0]:
    col = RED if yo[i] == 1 else BLUE
    ax.scatter(*Xo[i], c=col, s=70, edgecolor="k", lw=1.6, zorder=4)
    # point on own margin line (distance xi/||w|| along w)
    tgt = Xo[i] + yo[i] * wo / np.linalg.norm(wo) ** 2 * xi[i]
    ax.plot([Xo[i, 0], tgt[0]], [Xo[i, 1], tgt[1]], c="#8a2be2", lw=1.4, zorder=2)
big = np.argsort(-xi)[:3]
for i in big:
    ax.annotate(f"ξ={xi[i]:.1f}", Xo[i], xytext=(8, -14), textcoords="offset points", color="#8a2be2", fontsize=10, weight="bold")
ax.set(xlim=limo[:2], ylim=limo[2:], title="Soft margin: points that break the rule pay slack ξ (purple)")
ax.text(.1, -.35, "ξ=0 outside street · 0<ξ≤1 inside street, correct · ξ>1 wrong side", fontsize=8.5)
save(fig, "05_soft_margin_slack.png")

# ---- 6. hinge loss
fig, ax = plt.subplots(figsize=(7, 4.3))
t = np.linspace(-2, 3, 400)
ax.plot(t, np.maximum(0, 1 - t), c="#8a2be2", lw=3, label="hinge loss  max(0, 1 − y·f)")
ax.plot(t, (t < 0).astype(float), c="#bbb", lw=1.6, ls="--", label="0/1 loss (wrong side)")
ax.axvline(1, c="k", lw=.8, ls=":"); ax.axvline(0, c="k", lw=.8, ls=":")
ex = [(2.5, 0.0), (1.0, 0.0), (0.3, 0.7), (0.4, 0.6), (-1.2, 2.2)]
ax.scatter(*zip(*ex), c="#e67e22", s=70, zorder=5, label="the 5 points from the lecture's code")
ax.text(1.05, 2.4, "y·f ≥ 1\nno loss", fontsize=10); ax.text(0.02, 2.4, "inside\nstreet", fontsize=10)
ax.text(-1.95, 2.9, "wrong\nside", fontsize=10)
ax.set(xlabel="y · f(x)   (margin of the point)", ylabel="loss", xlim=(-2, 3), ylim=(-.2, 3.3),
       title="Hinge loss = the slack ξ")
ax.legend(fontsize=9, loc="upper right")
save(fig, "06_hinge_loss.png")

# ---- 7. effect of C
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, C in zip(axs, [0.01, 1, 100]):
    mc = SVC(kernel="linear", C=C).fit(Xo, yo)
    Zc = mc.decision_function(Go).reshape(ao.shape)
    ax.contourf(ao, bo, Zc, levels=[-1, 1], colors=[YEL], alpha=.55)
    ax.contour(ao, bo, Zc, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1.1, 2.1, 1.1])
    scatter2(ax, Xo, yo, s=32)
    ax.scatter(*mc.support_vectors_.T, s=120, fc="none", ec="k", lw=1.1, zorder=5)
    ax.set(xlim=limo[:2], ylim=limo[2:],
           title=f"C = {C}\nmargin = {2 / np.linalg.norm(mc.coef_):.2f} · {len(mc.support_)} support vectors")
fig.suptitle("Small C → wide tolerant margin (many SVs)      Large C → narrow strict margin (few SVs)", y=1.04, fontsize=13)
save(fig, "07_effect_of_C.png")

# ---------------------------------------------------------------- rings -> lift
fig = plt.figure(figsize=(12, 5))
Xr, yr = make_circles(n_samples=160, factor=.35, noise=.06, random_state=2)
yr = np.where(yr == 1, 1, -1)
ax = fig.add_subplot(1, 2, 1)
scatter2(ax, Xr, yr)
ax.set_aspect("equal"); ax.set_title("2-D: no straight line can separate the rings")
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(.62 * np.cos(th), .62 * np.sin(th), "k--", lw=1.6)
ax.text(0.05, -.7, "circle = plane in 3-D", fontsize=9, ha="left")
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
z = (Xr ** 2).sum(1)
ax3.scatter(Xr[yr == 1, 0], Xr[yr == 1, 1], z[yr == 1], c=RED, s=22)
ax3.scatter(Xr[yr == -1, 0], Xr[yr == -1, 1], z[yr == -1], c=BLUE, s=22)
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 10), np.linspace(-1.2, 1.2, 10))
ax3.plot_surface(gx, gy, np.full_like(gx, .38), alpha=.25, color="#2a9d4a")
ax3.set_title("Add z = x₁² + x₂²: a flat plane separates them")
ax3.set_xlabel("x₁"); ax3.set_ylabel("x₂"); ax3.set_zlabel("z"); ax3.view_init(18, -58)
save(fig, "08_lift_to_higher_dimension.png")

# ---- 9. kernel trick diagram
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)


def box(x, y, w_, h, txt, fc, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec="#444", lw=1.4))
    ax.text(x + w_ / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arr(x0, y0, x1, y1, col="#444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=col))


box(.2, 1.85, 1.9, .9, "x = (1, 2)\nz = (3, 1)", "#e8f1fb", bold=True)
# long route
arr(2.2, 2.5, 3.2, 3.5)
box(3.25, 3.1, 2.6, .9, "map to 3-D first\nφ(x)=(x₁², √2x₁x₂, x₂²)", "#fde2e0", 9.5)
arr(5.95, 3.55, 6.9, 3.55)
box(6.95, 3.1, 2.1, .9, "φ(x)·φ(z)\n= 9 + 12 + 4", "#fde2e0", 10)
ax.text(5.6, 4.35, "Slow way: build φ explicitly", fontsize=11, weight="bold", color="#b03a2e", ha="center")
# short route
arr(2.2, 2.1, 3.2, 1.1)
box(3.25, .65, 2.6, .9, "kernel function\nK(x,z) = (x·z)²", "#e1f5e4", 10, bold=True)
arr(5.95, 1.1, 6.9, 1.1)
box(6.95, .65, 2.1, .9, "5² = 25", "#e1f5e4", 11.5, bold=True)
ax.text(5.6, .2, "Kernel trick: same number, never leave 2-D", fontsize=11, weight="bold", color="#1e7d34", ha="center")
arr(9.15, 3.55, 9.9, 2.65, "#b03a2e"); arr(9.15, 1.1, 9.9, 2.0, "#1e7d34")
box(9.55, 2.05, 1.35, .6, "25  ✔", "#fff7d6", 12, bold=True)
ax.set_title("The Kernel Trick: the dot product in the big space, computed in the small one", fontsize=12.5)
save(fig, "09_kernel_trick.png")

# ---------------------------------------------------------------- moons
Xm, ym = make_moons(n_samples=200, noise=.2, random_state=0)
limm = (-2, 3.2, -1.5, 2.1)
am, bm, Gm = grid_for(limm, 250)


def moons_panel(ax, model, title):
    Zm = model.predict(Gm).reshape(am.shape)
    ax.contourf(am, bm, Zm, levels=[-.5, .5, 1.5], colors=[BLUE, RED], alpha=.22)
    ax.contour(am, bm, Zm, levels=[.5], colors="k", linewidths=1.6)
    ax.scatter(Xm[ym == 0, 0], Xm[ym == 0, 1], c=BLUE, s=16, edgecolor="white", lw=.4)
    ax.scatter(Xm[ym == 1, 0], Xm[ym == 1, 1], c=RED, s=16, edgecolor="white", lw=.4)
    ax.set_title(title, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])


# ---- 10. kernels compared
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, k in zip(axs, ["linear", "poly", "rbf"]):
    mk = SVC(kernel=k).fit(Xm, ym)
    moons_panel(ax, mk, f'kernel="{k}"   train acc = {mk.score(Xm, ym):.2f}')
save(fig, "10_kernels_compared.png")

# ---- 11. RBF similarity curve
fig, ax = plt.subplots(figsize=(7, 4.2))
d2 = np.linspace(0, 12, 300)
for g, col in [(0.05, "#2a9d4a"), (0.5, "#e67e22"), (5, "#c0392b")]:
    ax.plot(d2, np.exp(-g * d2), c=col, lw=2.6, label=f"γ = {g}")
ax.scatter([5, 5], [np.exp(-2.5), np.exp(-.25)], c="k", zorder=5)
ax.annotate("0.082", (5, np.exp(-2.5)), xytext=(6, .17), arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.annotate("0.779", (5, np.exp(-.25)), xytext=(6, .9), arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.axvline(5, c="#999", ls=":", lw=1)
ax.set(xlabel="squared distance  ‖x − z‖²", ylabel="K(x, z) = exp(−γ‖x−z‖²)", ylim=(0, 1.05),
       title="RBF kernel = similarity that fades with distance")
ax.legend()
save(fig, "11_rbf_similarity.png")

# ---- 12. gamma in pictures
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, g, lab in zip(axs, [0.1, 1, 50], ["too smooth (underfit)", "just right", "islands (overfit)"]):
    mg = SVC(kernel="rbf", C=1, gamma=g).fit(Xm, ym)
    moons_panel(ax, mg, f"γ = {g}  –  {lab}\ntrain acc = {mg.score(Xm, ym):.2f}")
save(fig, "12_gamma_in_pictures.png")

# ---- 13. C x gamma grid
fig, axs = plt.subplots(3, 3, figsize=(10.5, 10))
Cs, gs = [0.1, 1, 100], [0.1, 1, 50]
for i, C in enumerate(Cs):
    for j, g in enumerate(gs):
        mg = SVC(kernel="rbf", C=C, gamma=g).fit(Xm, ym)
        moons_panel(axs[i, j], mg, f"C = {C},  γ = {g}\ntrain acc = {mg.score(Xm, ym):.2f}")
fig.suptitle("Tune C (rows) and γ (columns) together", y=.94, fontsize=14)
save(fig, "13_C_and_gamma_grid.png")

# ---- 0. big-picture roadmap
fig, ax = plt.subplots(figsize=(12, 2.8)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 2.8)
steps = [("1  Lines & Planes", "hyperplane\nw·x + b = 0", "#e8f1fb"),
         ("2  Largest Margin", "widest street\nsupport vectors", "#e1f5e4"),
         ("3  Slack & Penalty", "soft margin\nhinge loss, C", "#fff3cd"),
         ("4  Lines to Curves", "kernel trick\nRBF, γ", "#fde2e0")]
for k, (t, s, c) in enumerate(steps):
    x0 = .2 + k * 3
    box(x0, .5, 2.4, 1.8, "", c)
    ax.text(x0 + 1.2, 1.85, t, ha="center", weight="bold", fontsize=11)
    ax.text(x0 + 1.2, 1.1, s, ha="center", fontsize=10.5)
    if k < 3: arr(x0 + 2.5, 1.4, x0 + 3.05, 1.4)
ax.set_title("Lecture roadmap", fontsize=13)
save(fig, "00_roadmap.png")

print("done")
