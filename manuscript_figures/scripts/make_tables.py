"""LaTeX tables for the manuscript, generated only from the saved result tables in results/tables."""
import numpy as np, pandas as pd
from pathlib import Path
T = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT = Path(__file__).resolve().parents[1] / "tables"; OUT.mkdir(parents=True, exist_ok=True)
CL = ["NORM", "MI", "STTC", "CD", "HYP"]
f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
f4 = lambda v: "--" if pd.isna(v) else f"{v:.4f}"
pct = lambda v: "--" if pd.isna(v) else f"{100*v:.1f}"
MOD = {"logreg": "Logistic regression (features)", "hgb_ens": "Gradient boosting (features)", "fcn_ens": "FCN ensemble",
       "resnet1d_ens": "ResNet1d ensemble (proposed)", "prevalence": "Prevalence"}

import re as _re
def write(name, s):
    s = _re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\begin{adjustbox}{max width=\\linewidth}\n\1", s, count=1)
    s = s.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
    s = s.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    s = s.replace("\\begin{table}[t]", "\\begin{table}[!t]").replace("\\begin{table}[!htbp]", "\\begin{table}[!t]")
    (OUT / f"{name}.tex").write_text(s)

# ---- Table 1: cohorts ----
sp = pd.read_csv(T / "T00_split_table.csv").set_index("split")
ds = pd.read_csv(T / "T01_dataset_summary.csv").set_index("dataset")
rows = []
for k, lab in [("train", "PTB-XL folds 1--8 (training)"), ("tune", "PTB-XL fold 9a (tuning)"),
               ("calib", "PTB-XL fold 9b (conformal calibration)"), ("test", "PTB-XL fold 10 (test)")]:
    r = sp.loc[k]
    rows.append(f"{lab} & {int(r.records):,} & {int(r.patients):,} & " + " & ".join(f"{int(r[f'pos_{c}']):,}" for c in CL) + r" \\")
for k, lab in [("georgia", "Georgia (external)"), ("chapman", "Chapman-Shaoxing (external)")]:
    r = ds.loc[k]; n = int(r.records)
    rows.append(f"{lab} & {n:,} & -- & " + " & ".join(f"{int(round(r[f'prev_{c}']*n)):,}" for c in CL) + r" \\")
write("tab2_cohorts", r"""\begin{table}[!htbp]
\centering\small
\caption{Cohorts after label harmonisation. Label columns give the number of ECGs positive for each diagnostic superclass (an ECG can carry more than one label). The four PTB-XL partitions are patient-disjoint. External cohorts are uniform random samples of 5,000 records per site (seed 2021) after exclusion of records without a superclass-mappable diagnosis or shorter than 10\,s; patient identifiers are not available for them.}
\label{tab:cohorts}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
Cohort & ECGs & Patients & NORM & MI & STTC & CD & HYP \\
\midrule
""" + "\n".join(rows[:4]) + "\n\\midrule\n" + "\n".join(rows[4:]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table 2: discrimination and calibration ----
t2 = pd.read_csv(T / "T02_discrimination_indomain.csv").set_index("model")
t3 = pd.read_csv(T / "T03_calibration_indomain.csv")
t8 = pd.read_csv(T / "T08_external_discrimination_calibration.csv")
rows = []
for m in ["logreg", "hgb_ens", "fcn_ens", "resnet1d_ens"]:
    r = t2.loc[m]
    ece = t3[(t3.model == m) & (t3.calibration == "temperature")].iloc[0]
    ext = []
    for site in ["georgia", "chapman"]:
        e = t8[(t8.site == site) & (t8.model == m)].iloc[0]
        ext.append(f"{f3(e.macro_auroc)} [{f3(e.macro_auroc_ci_low)}, {f3(e.macro_auroc_ci_high)}]")
    rows.append(f"{MOD[m]} & {f3(r.macro_auroc)} [{f3(r.macro_auroc_ci_low)}, {f3(r.macro_auroc_ci_high)}] & {f3(r.macro_auprc)} & "
                f"{f3(r.macro_f1)} & {f3(ece.macro_ece)} & {f3(ece.brier)} & " + " & ".join(ext) + r" \\")
write("tab3_discrimination", r"""\begin{table*}[t]
\centering\small
\caption{Discrimination and calibration on the untouched PTB-XL test fold and at the two external sites. Macro-averages over the five superclasses (PTB-XL) or over the evaluable superclasses (Georgia: NORM, STTC, CD, HYP; Chapman-Shaoxing: NORM, STTC, CD). 95\% confidence intervals from 1,000 patient-cluster (PTB-XL) or record-level (external) bootstrap replicates. ECE, classwise expected calibration error after per-label temperature scaling (15 equal-mass bins). F1 uses per-label thresholds chosen on the tuning partition.}
\label{tab:discrimination}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc|cc}
\toprule
 & \multicolumn{6}{c|}{PTB-XL fold 10} & \multicolumn{2}{c}{External macro-AUROC} \\
Model & Macro-AUROC & AUPRC & F1 & ECE & Brier & & Georgia & Chapman-Shaoxing \\
\midrule
""".replace(" & Brier & & ", " & Brier & ").replace(r"\begin{tabular}{lcccccc|cc}", r"\begin{tabular}{lccccc|cc}").replace(r"\multicolumn{6}{c|}{PTB-XL fold 10}", r"\multicolumn{5}{c|}{PTB-XL fold 10}")
      + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table 3: in-domain conformal ----
t5 = pd.read_csv(T / "T05_conformal_indomain.csv"); t5 = t5[(t5.model == "resnet1d_ens") & (t5.alpha == 0.10)]
t5b = pd.read_csv(T / "T05b_conformal_record_level.csv")
rows = []
for c in CL:
    for tc, lab in [(1, "positive"), (0, "negative")]:
        m = t5[(t5.cp == "mondrian") & (t5["class"] == c) & (t5.true_class == tc)].iloc[0]
        g = t5[(t5.cp == "global") & (t5["class"] == c) & (t5.true_class == tc)].iloc[0]
        first = c if tc == 1 else ""
        rows.append(f"{first} & {lab} & {int(m.n):,} & {f3(m.coverage)} [{f3(m.cov_ci_low)}, {f3(m.cov_ci_high)}] & "
                    f"{f3(g.coverage)} [{f3(g.cov_ci_low)}, {f3(g.cov_ci_high)}] & " + (pct(m.uncertain_rate) if tc == 1 else "") + r" \\")
rm = t5b[(t5b.model == "resnet1d_ens") & (t5b.cp == "mondrian") & (t5b.alpha == 0.10)].iloc[0]
rg = t5b[(t5b.model == "resnet1d_ens") & (t5b.cp == "global") & (t5b.alpha == 0.10)].iloc[0]
write("tab4_conformal", r"""\begin{table}[!htbp]
\centering\small
\caption{Coverage of conformal prediction sets on PTB-XL fold 10 at $\alpha=0.10$, stratified by label and true class (exact 95\% Clopper--Pearson intervals). Positive: label present (for NORM, a normal ECG); negative: label absent (for NORM, an abnormal ECG). Global CP uses one threshold per label; label-conditional CP uses one threshold per label and true class. The last column gives the share of test ECGs whose set for that label is \{0,1\} (label-conditional CP).}
\label{tab:conformal}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llrccc}
\toprule
Label & True class & $n$ & Label-conditional CP & Global CP & Uncertain (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{3}{l}{ECGs with $\ge$1 label referred (\%)} & """ + f"{pct(rm.referral_rate)} & {pct(rg.referral_rate)} & " + r"""\\
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table 4: external adaptation ----
t9 = pd.read_csv(T / "T09_external_conformal_summary.csv"); t9 = t9[t9.alpha == 0.10]
s6 = pd.read_csv(T / "S06_external_conformal_all_repeats.csv"); s6 = s6[(s6.alpha == 0.10) & (s6.true_class == 1)]
wr = s6.groupby(["site", "method", "n_adapt", "repeat"]).coverage.min().reset_index()
wr_p05 = wr.groupby(["site", "method", "n_adapt"]).coverage.quantile(0.05)
M = [("M0_source", "M0 source thresholds"), ("M2_weighted", "M2 covariate-shift weighting"), ("M4_pooled", "M4 pooled"),
     ("M3_hybrid", "M3 hybrid"), ("M1_target", "M1 target recalibration")]
rows = []
for site, sl in [("georgia", "Georgia"), ("chapman", "Chapman-Shaoxing")]:
    for i, (m, ml) in enumerate(M):
        cells = []
        for n in [100, 250, 1000]:
            s = t9[(t9.site == site) & (t9.method == m) & (t9.n_adapt == n)]
            pos = s[s.true_class == 1]; neg = s[s.true_class == 0]
            cells.append(f"{f3(pos.coverage_mean.min())} ({f3(wr_p05[(site, m, n)])})")
            if n == 250:
                neg250 = f3(neg.coverage_mean.min()); ref250 = pct(s.referral_rate_mean.mean())
        rows.append(f"{sl if i == 0 else ''} & {ml} & " + " & ".join(cells) + f" & {neg250} & {ref250}" + r" \\")
    rows.append(r"\midrule" if site == "georgia" else "")
write("tab6_external", r"""\begin{table*}[t]
\centering\small
\caption{Label-conditional conformal coverage at the external sites ($\alpha=0.10$; 20 random repeats per budget). Positive-class coverage: lowest label-wise mean coverage over repeats; in parentheses, the 5th percentile across repeats of the lowest label coverage within each repeat. $n$ is the number of labelled local ECGs available to M1, M3 and M4; M0 and M2 do not use local labels (M2 uses 947 unlabelled Georgia and 802 unlabelled Chapman-Shaoxing ECGs). Referral: share of evaluation ECGs with at least one label set equal to \{0,1\} or empty.}
\label{tab:external}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{3}{c}{Worst-label coverage, positive class} & Worst-label coverage, & Referral (\%) \\
Site & Method & $n=100$ & $n=250$ & $n=1000$ & negative class ($n=250$) & ($n=250$) \\
\midrule
""" + "\n".join(r for r in rows if r) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table 5: ablations ----
t7 = pd.read_csv(T / "T07_ablations.csv")
lab = {"CardioTrust-UQ (full)": "Full framework", "- temperature scaling": "without temperature scaling",
       "- Mondrian (global CP)": "global instead of label-conditional CP", "- ensemble (single seed)": "single network (seed 0) instead of 5-seed ensemble",
       "ResNet1d -> FCN backbone": "FCN instead of ResNet1d backbone"}
rows = [f"{lab[r.variant]} & {f3(r.macro_auroc)} & {f3(r.macro_ece)} & {f3(r.min_pos_coverage)} & {f3(r.min_stratum_coverage)} & "
        f"{r.mean_set_size:.2f} & {pct(r.referral_rate)}" + r" \\" for r in t7.itertuples()]
write("tab5_ablations", r"""\begin{table}[!htbp]
\centering\small
\caption{Ablations on PTB-XL fold 10 ($\alpha=0.10$), changing one component at a time. Worst positive: lowest positive-class coverage over the five labels; worst stratum: lowest over all ten label $\times$ true-class strata. Set size: mean number of values in a label's prediction set.}
\label{tab:ablations}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Variant & AUROC & ECE & Worst positive & Worst stratum & Set size & Referral (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table 6: age-aware (post hoc) ----
t14 = pd.read_csv(T / "T14_age_aware_conformal.csv")
rows = []
for a in [0.10, 0.05]:
    for v, vl in [("class_mondrian", "Label-conditional"), ("age_aware_mondrian", "Label-conditional + age band")]:
        s = t14[(t14.alpha == a) & (t14.variant == v) & (t14.true_class >= 0) & (t14.n >= 30)]
        d = s.coverage - (1 - a)
        pos = s[s.true_class == 1]
        ref = t14[(t14.alpha == a) & (t14.variant == v) & (t14.age_band == "ALL")].iloc[0]
        rows.append(f"{a:.2f} & {vl} & {int((d.abs() <= 0.05).sum())}/{len(s)} & {int((d < -0.05).sum())} & {f3(d.abs().mean())} & "
                    f"{f3(pos.coverage.min())} & {f3(s.coverage.min())} & {pct(ref.referral_rate)}" + r" \\")
write("tab7_age", r"""\begin{table}[!htbp]
\centering\small
\caption{Coverage across age bands (post hoc extension; PTB-XL fold 10). Cells are age band $\times$ label $\times$ true class with at least 30 test ECGs; worst positive: lowest positive-class cell. Within $\pm$0.05: cells whose coverage lies within 0.05 of $1-\alpha$; below: cells more than 0.05 under target; MAD: mean absolute deviation from $1-\alpha$.}
\label{tab:age}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{clcccccc}
\toprule
$\alpha$ & Calibration & Within $\pm$0.05 & Below & MAD & Worst positive & Worst cell & Referral (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print(sorted(p.name for p in OUT.glob("*.tex")))
