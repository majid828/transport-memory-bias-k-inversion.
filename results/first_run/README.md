# First run (30 September 2026)

Tables, figures and run information of the first full run of `run_all.py`, kept unchanged.
These are the numbers quoted in the written proposal and on the Experiment B, Experiment C and
conclusions slides (slides 29, 30 and 32).

The first run used the same code and the same datasets as this repository, with older library versions:
Python 3.12, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.2, matplotlib 3.10.8.
The datasets are identical in both runs (bit for bit). The trained networks differ slightly between
scikit-learn versions, so `python run_all.py` with `requirements.txt` gives `results/main`, which differs
from these tables by small amounts (see the main README). To recreate these tables, install the versions
above; this was not re-tested, because those versions could not be installed in the environment used
to prepare this repository.
