import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta

# ── Setup ──────────────────────────────────────────────────────────────────────
N_OBS   = 20
PRIOR_A = 1
PRIOR_B = 1
Y_VALUES = [2, 6, 10, 14, 18]
COLORS   = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]

omega = np.linspace(0.001, 0.999, 500)

prior       = beta(PRIOR_A, PRIOR_B)
prior_mean  = prior.mean()
prior_var   = prior.var()

posteriors = []
for y, col in zip(Y_VALUES, COLORS):
    a, b    = PRIOR_A + y, PRIOR_B + N_OBS - y
    dist    = beta(a, b)
    posteriors.append(dict(y=y, dist=dist, mean=dist.mean(),
                           var=dist.var(), color=col, a=a, b=b))

avg_post_mean  = np.mean([p["mean"] for p in posteriors])
avg_post_var   = np.mean([p["var"]  for p in posteriors])
var_post_means = np.var( [p["mean"] for p in posteriors])

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
MUTED   = "#8b949e"
WHITE   = "#e6edf3"
GREEN   = "#2a9d8f"
PURPLE  = "#9b5de5"
ORANGE  = "#f4a261"

fig = plt.figure(figsize=(13, 16), facecolor=BG)
fig.suptitle("Bayesian Inference  ·  Prior → Posterior",
             fontsize=16, color=WHITE, y=0.98, fontfamily="serif")

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                       left=0.08, right=0.96, top=0.94, bottom=0.04)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=MUTED, fontsize=9,
                 loc="left", pad=8, fontfamily="monospace")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

# ── 1. Prior ───────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "PRIOR  ·  p(ω) = Beta(1,1) = Uniform(0,1)")
y_prior = prior.pdf(omega)
ax1.fill_between(omega, y_prior, alpha=0.25, color=GREEN)
ax1.plot(omega, y_prior, color=GREEN, lw=2)
ax1.axvline(prior_mean, color=GREEN, ls="--", lw=1.5,
            label=f"prior mean = {prior_mean:.2f}")
ax1.set_ylim(0, 2.5)
ax1.set_xlabel("ω  (probability of female birth)")
ax1.legend(fontsize=9, framealpha=0, labelcolor=GREEN)
ax1.text(0.5, 0.6,
         f"mean = {prior_mean:.3f}     variance = {prior_var:.4f}",
         transform=ax1.transAxes, ha="center", color=MUTED, fontsize=9)

# ── 2. Posteriors ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2, "POSTERIORS  ·  p(ω|y) = Beta(1+y,  1+n−y)   [n=20]")
ax2.axvline(prior_mean, color=GREEN, ls="--", lw=1, alpha=0.5,
            label="prior mean")
ax2.axvline(0.5, color=BORDER, ls=":", lw=1)

for p in posteriors:
    pdf = p["dist"].pdf(omega)
    ax2.plot(omega, pdf, color=p["color"], lw=2,
             label=f"y={p['y']}  mean={p['mean']:.3f}")
    ax2.axvline(p["mean"], color=p["color"], ls="--", lw=0.8, alpha=0.6)

ax2.set_xlabel("ω")
ax2.legend(fontsize=8.5, framealpha=0, ncol=3,
           labelcolor="white", loc="upper left")

# ── 3. Eq 2.7 — bar of posterior means ────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, "EQ 2.7  ·  E(ω) = E(E(ω|y))")
means = [p["mean"] for p in posteriors]
cols  = [p["color"] for p in posteriors]
bars  = ax3.bar([f"y={p['y']}" for p in posteriors], means,
                color=cols, alpha=0.8, width=0.6)
ax3.axhline(prior_mean, color=GREEN, ls="--", lw=1.5,
            label=f"prior mean = {prior_mean:.3f}")
ax3.axhline(avg_post_mean, color=WHITE, ls=":", lw=1.5,
            label=f"avg post. mean = {avg_post_mean:.3f}")
ax3.set_ylim(0, 1)
ax3.set_ylabel("Posterior mean")
ax3.legend(fontsize=8, framealpha=0)
ax3.text(0.5, -0.22,
         "avg posterior mean ≈ prior mean  ✓",
         transform=ax3.transAxes, ha="center", color=GREEN, fontsize=8.5)

# ── 4. Eq 2.8 — variance decomposition ────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4, "EQ 2.8  ·  var(ω) = E(var(ω|y)) + var(E(ω|y))")

labels  = ["E(var(ω|y))\navg post. var", "var(E(ω|y))\nvar of means", "Prior var"]
values  = [avg_post_var, var_post_means, prior_var]
bar_col = [PURPLE, ORANGE, GREEN]
x = np.arange(len(labels))
ax4.bar(x, values, color=bar_col, alpha=0.85, width=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8, color=WHITE)
for xi, v in zip(x, values):
    ax4.text(xi, v + 0.0002, f"{v:.5f}", ha="center", color=WHITE, fontsize=8)
ax4.set_ylabel("Variance")
ax4.text(0.5, -0.28,
         f"{avg_post_var:.5f} + {var_post_means:.5f} = {avg_post_var+var_post_means:.5f}  ≈  {prior_var:.5f}  ✓",
         transform=ax4.transAxes, ha="center", color=MUTED, fontsize=8)

# ── 5. Stacked variance bar ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, :])
style_ax(ax5, "VARIANCE DECOMPOSITION  ·  prior var split into two pieces")
pct1 = avg_post_var   / prior_var * 100
pct2 = var_post_means / prior_var * 100
ax5.barh(["Prior\nvariance"], [pct1], color=PURPLE, alpha=0.85,
         label=f"E(var(ω|y))  =  {avg_post_var:.5f}  ({pct1:.0f}%  — residual uncertainty)")
ax5.barh(["Prior\nvariance"], [pct2], left=[pct1], color=ORANGE, alpha=0.85,
         label=f"var(E(ω|y)) =  {var_post_means:.5f}  ({pct2:.0f}%  — data moves estimate)")
ax5.set_xlim(0, 105)
ax5.set_xlabel("% of prior variance")
ax5.legend(fontsize=9, framealpha=0, loc="lower right")
ax5.text(pct1/2,        0, f"{pct1:.0f}%", va="center", ha="center",
         color=WHITE, fontsize=11, fontweight="bold")
ax5.text(pct1 + pct2/2, 0, f"{pct2:.0f}%", va="center", ha="center",
         color=WHITE, fontsize=11, fontweight="bold")

plt.savefig("bayesian_viz.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved → bayesian_viz.png")
