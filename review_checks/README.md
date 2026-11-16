# Review checks

Robustness and mechanism checks run after the first experiment, with the same networks, seeds and
metrics. Run them from the repository root, after `python run_all.py` (which makes `data/main`).
`run_check.py --name baseline` must run before `review_table2.py`, which reads its `table_shift.csv`.

| Script | What it tests | Output (in `results/review_checks/`) |
|---|---|---|
| `run_check.py --name baseline` | the six main models on the main design (reproduction) | `baseline/table_methods.csv`, `table_shift.csv`, `per_case.csv`, `info.json` |
| `run_check.py --name porosity --theta 0.2 0.4` | mobile porosity unknown, drawn from U(0.2, 0.4) per aquifer | `porosity/` (same files) |
| `run_check.py --name domain64 --nx 64 --ny 32` | domain twice as long and wide (9–21 correlation lengths long) | `domain64/` (same files) |
| `ablation.py` | which observations fix the magnitude and the pattern of K (Experiment D, slide 31) | `ablation/ablation_table.csv` |
| `extra_checks.py` | equilibrium limit (Da = 100, 1000); spatially variable β | `baseline/extra_summary.csv`, `extra_per_case.csv` |
| `extra2.py` | equilibrium exchange vs a no-exchange aquifer with K/(1+β); variable β with snapshot data | `baseline/extra2_summary.csv`, `extra2_per_case.csv` |
| `extra3.py` | no-exchange aquifer with K/(1+β) and D/(1+β): tests (and rejects) the dispersion explanation | `baseline/extra3_summary.csv`, `extra3_per_case.csv` |
| `extra4.py` | clean-BTC comparison; BTC rescaled by (1+β); memory-blind model on area-normalized BTCs | `baseline/extra4_summary.csv`, `extra4_per_case.csv`, `extra4_btc_check.csv` |
| `extra5.py` | memory-agnostic and joint models on area-normalized BTCs (does the BTC level act as a magnitude constraint?) | `baseline/extra5_summary.csv`, `extra5_per_case.csv` |
| `review_table2.py` | all rows of Table A1 (Appendix A of the chapter) in one run | `baseline/table2_summary.csv`, `table2_per_case.csv`, `table2_nomemory_accuracy.csv` |

The porosity and domain64 designs generate their own datasets in `data/porosity` and `data/domain64`.
`FHINV_WORKERS=2` runs the transport simulations of a dataset on two processes (identical results).
