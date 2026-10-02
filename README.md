# Label-conditional conformal prediction for multi-label 12-lead ECG diagnosis

Code and results for the manuscript *"Label-conditional conformal prediction for multi-label 12-lead ECG diagnosis: protecting diseased patients and restoring coverage at external hospitals"* (R. Augustian Isaac, Amreen Ayesha, A. Jaffar Sadiq Ali, S. Mercy Gnana Gandhi; submitted to *Computer Methods and Programs in Biomedicine*).

The study trains a five-seed ResNet1d ensemble on PTB-XL, adds per-label temperature scaling and label-conditional (Mondrian) conformal prediction, and measures how coverage for label-positive ECGs changes at two external hospitals (Georgia and Chapman-Shaoxing, PhysioNet/CinC Challenge 2021) under five site-adaptation strategies.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/ecg_label_conditional_conformal.ipynb` | Complete pipeline: data download and checksum verification, preprocessing, models, calibration, conformal prediction, external adaptation, robustness, post hoc analyses, tables and figures |
| `results/tables/` | All result tables from the full run (T00–T16, S01–S08) as CSV |
| `results/config/` | Run configuration, normalisation statistics, environment lock and run manifest |
| `results/data_audit/DATA_PROVENANCE.json` | Data sources, versions, DOIs and checksum verification record |
| `manuscript_figures/scripts/` | Scripts that rebuild the manuscript figures and LaTeX tables (Tables 2–7) from `results/tables/`; Table 1 (comparison with previous studies) is a literature table and is provided directly |
| `manuscript_figures/figures/`, `manuscript_figures/tables/` | The resulting figures (PDF, PNG) and tables (LaTeX) |

## Data

No data are redistributed here. The notebook downloads the public datasets directly from PhysioNet and verifies them against the published SHA-256 checksums:

- PTB-XL v1.0.3 — https://doi.org/10.13026/kfzx-aw45
- PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3 — https://doi.org/10.13026/34va-7q14

Both are released under the Creative Commons Attribution 4.0 licence.

## Reproducing the results

1. Open `notebooks/ecg_label_conditional_conformal.ipynb` in Google Colab (GPU runtime, e.g. T4) or a local Jupyter environment with a CUDA GPU.
2. In the configuration cell, set `MODE = "FULL_PAPER_RUN"` (use `"SMOKE_TEST"` for a quick end-to-end check).
3. Run all cells. The full run takes several hours on a T4 GPU, mostly for data download and the ten deep models. Intermediate files are cached, so an interrupted run can be resumed.
4. The notebook writes all tables, figures and a results ZIP to its output folder.

To rebuild only the manuscript figures and tables from the saved CSV files:

```bash
pip install -r requirements.txt
python manuscript_figures/scripts/make_figures.py
python manuscript_figures/scripts/make_tables.py
```

## Notes on the result files

- **Predeclared success criteria.** Before the final run, eight pass/fail criteria (G0–G7) were fixed in the notebook; the outcome is in `results/tables/T13_gate_dashboard.csv`. Six were met. In-domain stratum coverage (G3: 0.876 against 0.88) and subgroup coverage (G4: 0.676 against 0.85) were not, as reported in the article. The notebook maps any failed G3 to the automatic label `STOP-REDESIGN` in `results/config/run_manifest.json`; that rule was intended to catch an invalid conformal implementation. The post hoc exact finite-sample check (`T16`) and 200 calibration re-splits (`T15`) examine this directly and are reported as post hoc in the article.
- **Working title.** The analysis was run under the working title "CardioTrust-UQ"; that name survives in a few output labels (for example the first row of `T07_ablations.csv`). The code has since been renamed; results are unchanged.
- **Output inventory.** `S99_output_inventory.csv` lists every file the run produced, including an internal results checklist (`RESULTS_AUDIT.md`) and model checkpoints that are not distributed here.

## Software

Python 3.13 with the package versions in `requirements.txt` (exact versions used for the reported run are in `results/config/environment_lock.json`).

## Citation

If you use this code, please cite the article (details will be added after publication) and this software archive (see `CITATION.cff`).

## Licence

Code: MIT licence (see `LICENSE`). The datasets remain under their original PhysioNet licences.
