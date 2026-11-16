# Transport-memory bias in learned hydraulic-conductivity inversion

Code, data and results for **Chapter 3** of the PhD dissertation
*Identifying and Representing Hydrologic Memory: From Recurrent-State Neural Networks to Mechanism-Aware
Subsurface Transport Inversion*. Author: Majid Hussain Shah, Department of Geological Sciences,
The University of Alabama.

**Question.** A neural network that estimates hydraulic conductivity *K* from heads and tracer data
learns only the physics of its training simulator. If that simulator has no solute storage
(mobile–immobile exchange, "memory"), how does storage in the data bias the *K* estimate, in its
magnitude (mean ln *K*) or in its pattern, and which training data and observations remove the bias?

**Approach.** Synthetic 2-D aquifers in which the true *K* is known. Each test aquifer is simulated
seven times with the **same** true *K*: once without memory and with three storage capacities
β = 0.25, 0.5, 1, each with fast (Da = 30) or slow (Da = 1) exchange. Any change in the estimated
*K* is therefore caused by memory alone (or by noise, which is measured separately).

## Repository layout

```
run_all.py                 main experiment (Experiments A-C, controls, figures)
fhinv/                     the package
  sim.py                   K fields, steady flow, mobile-immobile transport, observations, noise
  data.py                  training and test datasets
  learn.py                 PCA representation of ln K, the neural networks, metrics, forward check
  controls.py              noise floor of the pattern metric; shrinkage of each estimator
  plots.py                 figures of the main experiment
review_checks/             robustness and mechanism checks, incl. Experiment D (see review_checks/README.md)
slide_figures/             scripts for the figures on the Chapter 3 slides; output/ holds the PNGs and statistics
slide_numbers.py           recomputes every number on the Chapter 3 slides from the stored results
PREDICTIONS.md             predictions written before the first run
data/
  main/                    train_mixed, train_blind, test_controlled, test_random (.npz)
  porosity/, domain64/     datasets of two review checks
results/
  main/                    run_all.py output with this code and requirements.txt (reproducible)
  first_run/               first full run (older library versions); numbers in the written proposal
  review_checks/           baseline, porosity, domain64, ablation
  slide_numbers.csv        output of slide_numbers.py
```

## Installation

Tested with Python 3.13 on Linux, one CPU core, no GPU.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --quick            # smoke test, about 1 minute (data/quick, results/quick)
```

## Reproducing the results

Run from the repository root, in this order. Times are for one CPU core.

```bash
python run_all.py                                                # about 5 min (makes data/main, results/main)
python review_checks/run_check.py --name baseline                # under 1 min
python review_checks/ablation.py                                 # about 1 min (Experiment D)
python review_checks/run_check.py --name porosity --theta 0.2 0.4    # about 3 min
python review_checks/run_check.py --name domain64 --nx 64 --ny 32   # about 35 min (FHINV_WORKERS=2 uses two cores)
python review_checks/extra_checks.py                             # the extra checks: about 4 min in total
python review_checks/extra2.py
python review_checks/extra3.py
python review_checks/extra4.py
python review_checks/extra5.py
python review_checks/review_table2.py
python slide_figures/make_slide_figures.py
python slide_figures/fig_btc_storage.py
python slide_figures/fig_experiment_B.py
python slide_numbers.py
```

Datasets are generated only if they are missing, so with the stored `data/` folder the scripts go
straight to training. `FHINV_WORKERS=2` runs the transport simulations on two processes (same results). All random numbers come from fixed seeds: the datasets, the noise and the
network initializations are the same in every run.

## Which result is on which slide

Slide numbers refer to the Chapter 3 slides of the proposal-defense presentation.

| Slide | Result | Produced by | Stored in |
|---|---|---|---|
| 24, 27, 28, 81, 88 | field maps, design, Experiment A (one field, three memory conditions), heads vs 2K | `slide_figures/make_slide_figures.py` | `slide_figures/output/` |
| 25 | slow flow or storage: computed 1-D breakthrough curves | `slide_figures/fig_btc_storage.py` | `slide_figures/output/` |
| 29 | Experiment B: memory-blind network, magnitude and pattern | `run_all.py` → `slide_figures/fig_experiment_B.py` | `results/first_run/tables/` |
| 30 | Experiment C: training with memory | `run_all.py` | `results/first_run/tables/table_methods_controlled.csv` |
| 31 | Experiment D: which observations help | `review_checks/ablation.py`, `run_check.py --name porosity` | `results/review_checks/ablation/`, `porosity/` |
| 32 | conclusions, forward check | `run_all.py` | `results/first_run/tables/` |
| notes of 29 | Da = 1000 and area-normalized curves (Table A1) | `review_checks/review_table2.py` | `results/review_checks/baseline/table2_summary.csv` |

`python slide_numbers.py` prints every number on these slides next to the value recomputed from the
stored results and its source file (67 numbers; all match).

## First run and reproducible re-run

The first full run (30 September 2026) used Python 3.12 with numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0, pandas 3.0.2 and matplotlib 3.10.8. Its tables are kept unchanged in
`results/first_run/` (see `results/first_run/README.md`); the written proposal and slides 29, 30 and 32
quote them. Everything else in
`results/` was produced with this code and the versions in `requirements.txt`, and running the
commands above with those versions reproduces it.

The datasets are identical in both runs, bit for bit. The trained networks differ slightly between
scikit-learn versions: across all methods and test conditions, `results/main` differs from
`results/first_run` by at most 0.011 in the error of mean ln K, 0.004 in the pattern error, 0.021 in the β error
and 0.021 in the memory-induced shift of mean ln K. The conclusions are the same. Because of this, some numbers in
`results/main` differ from the slides in the second decimal (for example, *K* 10–17 % too high
instead of 10–15 % for the memory-trained networks without memory).

## Model summary

* **Aquifers.** ln *K* = μ + σ z(**x**) on a 32 × 16 grid, z a standardized Gaussian field with covariance
  exp(−r²/ℓ²); μ ~ N(0, 0.5²), σ ~ U(0.5, 1), ℓ ~ U(3, 7) cells, so the magnitude of *K* varies between aquifers.
* **Flow.** Steady, fixed heads left (h = 1) and right (h = 0), no-flow top and bottom. With these
  boundaries, scaling *K* by a constant leaves the heads unchanged and scales the discharge *Q*.
* **Transport.** Single-rate mobile–immobile model (van Genuchten & Wierenga, 1976): capacity
  β = θim/θm, exchange rate α, Da = α T_ref; mobile porosity 0.30; dispersion 0.05.
* **Observations.** Heads at 12 wells (noise 0.005), outlet breakthrough curve at 48 times between
  0.05 and 6 T_ref, one plume snapshot at 24 points at 0.75 T_ref, and the discharge *Q* (5 % noise).
* **Data.** 2,400 training aquifers with random memory (25 % none; β ~ U(0.1, 1.5); Da log-uniform
  in 0.3–100); the same aquifers without memory (training set of the memory-blind networks);
  60 new test fields × 7 memory conditions; 400 in-distribution test aquifers.
* **Networks.** scikit-learn `MLPRegressor`, 2 × 128 ReLU units, early stopping, five seeds.
  ln *K* is represented by its mean plus 40 principal components (97 % of the pattern variance).
  Memory-blind (trained without memory), memory-agnostic (trained with memory, predicts *K*) and
  joint (trained with memory, also predicts β and Da) networks, with different sets of observations.
* **Metrics.** Shift of the estimated mean ln *K* for the same field; RMSE of mean ln *K*; RMSE of the
  zero-mean pattern; β error; noise floor of the pattern change; forward re-simulation misfit.

## Data format

Each `.npz` file holds the true fields `lnk` (N × 16 × 32), `mu` (mean ln *K*), `beta`, `logda`
(log10 Da) and `theta` (mobile porosity); the noisy observations `obs_heads`, `obs_btc`, `obs_snap`,
`obs_Q`; and the noise-free values `clean_*`. `test_controlled.npz` also has `cond` (0–6, see
`fhinv/data.py` `CONDITIONS`) and `field` (0–59). Load with `numpy.load(path)`.

## Scope

Two-dimensional Gaussian fields, steady fixed-head flow, single-rate uniform exchange (spatially
variable β in one review check), known porosity and injected mass (unknown porosity in one review
check), networks tested on data from their own simulator, a reduced-order representation instead of a
convolutional network, and no field data. The results test the failure mode under controlled
assumptions; they do not establish performance on real aquifers.
