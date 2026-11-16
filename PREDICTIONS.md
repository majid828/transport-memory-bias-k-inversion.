# Pre-specified predictions (written before the experiment was run)

Setting: 2-D synthetic aquifers, fixed-head flow, conservative tracer, single-rate
mobile-immobile exchange with capacity ratio beta and nominal Damkohler number
Da = alpha * T_ref.  Learned inverse models map observations to ln K.

P1  Equilibrium regime (Da = 30).  A memory-blind inverse (trained on beta = 0 data only)
    applied to the SAME true K fields with increasing beta will shift the estimated mean
    ln K downward by approximately ln(1 + beta) (0.22, 0.41, 0.69 for beta = 0.25, 0.5, 1.0),
    while the estimated spatial pattern of ln K changes comparatively little.
    Reason: with fixed-head boundaries, heads do not depend on the magnitude of K, and fast
    exchange acts as a uniform retardation R = 1 + beta, which is equivalent in travel time
    to dividing K by R.

P2  Non-equilibrium regime (Da = 1).  The mean-ln K shift will be smaller than ln(1 + beta)
    (the peak is barely delayed), but the change in the inferred spatial pattern (relative to
    the beta = 0 estimate of the same field) will be larger than in the equilibrium regime,
    i.e. tailing is converted into spurious spatial structure ("false heterogeneity").

P3  Identifiability limit.  With heads + outlet BTC only, models trained on data that include
    memory (memory-agnostic or joint) will reduce the mean-ln K bias in the non-equilibrium
    regime but NOT reliably in the equilibrium regime, where only K/(1 + beta) is constrained.
    Adding an independent magnitude constraint (total discharge Q) will reduce the
    equilibrium-regime mean-ln K error.

P4  Fair control.  The joint model (outputs K and memory parameters) will NOT be clearly
    better than a memory-agnostic model trained on the same data with the same inputs but
    K-only outputs: most of the benefit comes from the training distribution and the
    information content of the observations, not from the extra output branch.

P5  Heads alone are unaffected by memory (steady flow) but cannot estimate mean ln K.

Any of these can fail; results are reported as obtained.
written 2026-09-30T04:29:11Z
