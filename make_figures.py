"""Publication figures, generated only from the saved result tables in results/tables."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

T = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(parents=True, exist_ok=True)
OK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.linewidth": 0.8})
CL = ["NORM", "MI", "STTC", "CD", "HYP"]
SITE = {"georgia": "Georgia (USA)", "chapman": "Chapman-Shaoxing (China)"}
METH = {"M0_source": "M0 source thresholds", "M1_target": "M1 target recalibration",
        "M2_weighted": "M2 covariate-shift weighting", "M3_hybrid": "M3 hybrid", "M4_pooled": "M4 pooled"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

# ---------- Fig 1: framework ----------
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=6.8, weight="bold")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=6.6, linespacing=1.15)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333"))
G, B, Y = "#EFEFEF", "#DCEBF7", "#FBEBD3"
box(0.5, 30, 18, 13, "12-lead ECG", "10 s, 100 Hz\n0.5 Hz high-pass", G)
box(23, 30, 19.5, 13, "ResNet1d ensemble", "5 seeds, logit average\nfolds 1-8 (training)", G)
box(47, 30, 19.5, 13, "Temperature scaling", "per label; fitted on\nfold 9 tuning half", B)
box(71, 30, 28, 13, "Label-conditional CP", "threshold per label x true class\nfold 9 calibration half", B)
for a, b in [(19.3, 22.2), (43.3, 46.2), (67.3, 70.2)]:
    arrow(a, 36.5, b, 36.5)
box(62, 11, 11, 11, "Present", "{1}", "#E3F2E1"); box(75, 11, 11, 11, "Absent", "{0}", "#E3F2E1")
box(88, 11, 11, 11, "Refer", "{0,1} or {}", "#F8DCDC")
for xx in (67.5, 80.5, 93.5):
    arrow(85, 29.4, xx, 22.8)
ax.text(62, 5.5, "Guarantee per label and true class:  P(y in C | y = c) >= 1 - alpha", fontsize=6.8, style="italic")
box(1, 5, 26, 18, "New hospital", "Georgia / Chapman-Shaoxing\nfrozen model and temperature", Y)
box(31, 5, 27, 18, "Site recalibration", "M1: labelled local ECGs\n(n = 100-1000)\nM2: unlabelled weighting\nM3 hybrid, M4 pooled", Y)
arrow(27.6, 14, 30.6, 14); arrow(58.6, 14, 61.6, 16.5)
ax.plot([83, 83], [29.8, 29.8])
save(fig, "Fig1_framework")

# ---------- Fig 2: in-domain Mondrian vs global ----------
t5 = pd.read_csv(T / "T05_conformal_indomain.csv")
t5 = t5[(t5.model == "resnet1d_ens") & (t5.alpha == 0.10)]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
keys = [(c, t) for c in CL for t in (1, 0)]; x = np.arange(len(keys)); w = 0.38
for k, (v, col, lab) in enumerate([("mondrian", OK[0], "Label-conditional (Mondrian) CP"), ("global", OK[1], "Global per-label CP")]):
    s = t5[t5.cp == v].set_index(["class", "true_class"])
    cov = np.array([s.loc[kk, "coverage"] for kk in keys])
    lo = cov - np.array([s.loc[kk, "cov_ci_low"] for kk in keys]); hi = np.array([s.loc[kk, "cov_ci_high"] for kk in keys]) - cov
    ax.bar(x + (k - 0.5) * w, cov, w, yerr=[lo, hi], color=col, label=lab, capsize=1.8, error_kw={"lw": 0.7})
ax.axhline(0.90, color="k", ls="--", lw=0.9, label="Target 0.90")
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{'positive' if t else 'negative'}" for c, t in keys], fontsize=7)
ax.set_ylim(0.35, 1.02); ax.set_ylabel("Coverage (95% CI)")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
save(fig, "Fig2_indomain_coverage")

# ---------- Fig 3: external adaptation (coverage + referral) ----------
t9 = pd.read_csv(T / "T09_external_conformal_summary.csv"); t9 = t9[t9.alpha == 0.10]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
styles = ["-", "--", "-.", ":", (0, (5, 1))]; marks = ["o", "s", "^", "D", "v"]
for col, site in enumerate(["georgia", "chapman"]):
    s = t9[t9.site == site]
    cov = s[s.true_class == 1].groupby(["method", "n_adapt"]).coverage_mean.min().reset_index()
    s6 = pd.read_csv(T / "S06_external_conformal_all_repeats.csv")
    s6 = s6[(s6.alpha == 0.10) & (s6.true_class == 1) & (s6.site == site)]
    p05 = (s6.groupby(["method", "n_adapt", "repeat"]).coverage.min().reset_index()
           .groupby(["method", "n_adapt"]).coverage.quantile(0.05).rename("coverage_p05").reset_index())
    ref = s.groupby(["method", "n_adapt"]).referral_rate_mean.mean().reset_index()
    for k, m in enumerate(METH):
        c = cov[cov.method == m]
        axes[0, col].plot(c.n_adapt, c.coverage_mean, linestyle=styles[k], marker=marks[k], color=OK[k], ms=4.2 - 0.4 * k, lw=1.3, label=METH[m])
        if m == "M1_target":
            p = p05[p05.method == m]
            axes[0, col].fill_between(p.n_adapt, p.coverage_p05, c.coverage_mean.values, color=OK[k], alpha=0.15, lw=0)
        r = ref[ref.method == m]
        axes[1, col].plot(r.n_adapt, r.referral_rate_mean, linestyle=styles[k], marker=marks[k], color=OK[k], ms=4.2 - 0.4 * k, lw=1.3)
    axes[0, col].axhline(0.90, color="k", ls="--", lw=0.8)
    axes[0, col].set_title(SITE[site], fontsize=8.5)
    axes[1, col].set_xlabel("Labelled local ECGs used for recalibration")
    for a in axes[:, col]:
        a.set_xscale("log"); a.set_xticks([100, 250, 500, 1000]); a.set_xticklabels(["100", "250", "500", "1000"])
axes[0, 0].set_ylabel("Worst-label coverage,\nlabel-positive ECGs"); axes[1, 0].set_ylabel("ECGs referred for review")
h, l = axes[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
save(fig, "Fig4_external_adaptation")

# ---------- Fig 4: age-aware ----------
t14 = pd.read_csv(T / "T14_age_aware_conformal.csv")
t14 = t14[(t14.alpha == 0.10) & (t14.true_class == 1) & (t14.n >= 30)]
order = ["<40", "40-59", "60-74", ">=75"]
cells = sorted(t14[["age_band", "class"]].drop_duplicates().values.tolist(), key=lambda r: (order.index(r[0]), CL.index(r[1])))
fig, ax = plt.subplots(figsize=(7.2, 2.8)); x = np.arange(len(cells)); w = 0.38
for k, (v, col, lab) in enumerate([("class_mondrian", OK[0], "Label-conditional"), ("age_aware_mondrian", OK[2], "Label-conditional + age band")]):
    s = t14[t14.variant == v].set_index(["age_band", "class"])
    cov = np.array([s.loc[tuple(c), "coverage"] for c in cells])
    lo = cov - np.array([s.loc[tuple(c), "ci_low"] for c in cells]); hi = np.array([s.loc[tuple(c), "ci_high"] for c in cells]) - cov
    ax.bar(x + (k - 0.5) * w, cov, w, yerr=[lo, hi], color=col, label=lab, capsize=1.6, error_kw={"lw": 0.6})
ax.axhline(0.90, color="k", ls="--", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([f"{a.replace('>=', '≥')}\n{c}" for a, c in cells], fontsize=6.6)
ax.set_ylim(0.4, 1.02); ax.set_ylabel("Coverage, label-positive\nECGs (95% CI)")
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
save(fig, "Fig5_age_aware")

# ---------- Fig 5: calibration re-splits ----------
raw = pd.read_csv(T / "S07_calibration_resplits_raw.csv"); raw = raw[raw.alpha == 0.10]
t15 = pd.read_csv(T / "T15_calibration_resplits_summary.csv"); t15 = t15[t15.alpha == 0.10]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
data = [raw[(raw["class"] == c) & (raw.true_class == t)].coverage.values for c, t in keys]
ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": OK[0], "lw": 1.2}, boxprops={"lw": 0.8}, whiskerprops={"lw": 0.8})
orig = [t15[(t15["class"] == c) & (t15.true_class == t)].original_split.values[0] for c, t in keys]
ax.plot(np.arange(1, len(keys) + 1), orig, "D", color=OK[3], ms=3.5, label="Primary calibration split")
ax.axhline(0.90, color="k", ls="--", lw=0.9, label="Target 0.90")
ax.set_xticks(np.arange(1, len(keys) + 1)); ax.set_xticklabels([f"{c}\n{'positive' if t else 'negative'}" for c, t in keys], fontsize=7)
ax.set_ylabel("Fold-10 coverage across\n200 re-splits"); ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
save(fig, "Fig3_calibration_resplits")

# ---------- Supplementary: reliability not reproducible from tables (needs predictions); robustness ----------
t11 = pd.read_csv(T / "T11_robustness.csv")
names = {"clean": "Clean", "noise_20": "Noise 20 dB", "noise_10": "Noise 10 dB", "baseline_wander": "Baseline wander",
         "lead_loss": "One lead lost", "gain_0.5": "Gain x0.5"}
fig, ax = plt.subplots(figsize=(6.5, 2.6)); x = np.arange(len(t11))
ax.bar(x - 0.2, t11.macro_auroc, 0.4, color=OK[0], label="Macro-AUROC")
ax.bar(x + 0.2, t11.min_pos_coverage, 0.4, color=OK[2], label="Worst-label coverage, label-positive ECGs")
ax.axhline(0.90, color="k", ls="--", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([names[p] for p in t11.perturbation], fontsize=7); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7, loc="lower left")
save(fig, "FigS1_robustness")
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
