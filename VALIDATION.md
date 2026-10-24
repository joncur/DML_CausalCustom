# Validation record

## Recommendation extension

Current full-suite result: **34 passed in 4.26 seconds**. The 24 original checks
remain. Ten added checks cover categorical DR recovery, both double-robustness
branches on controlled fixtures, known experimental probabilities, multiarm
contrasts, W effect modification averaged over X, serialization, independent
policy value and clustered intervals, costs/support/fallbacks, calibration
separation, failed refits, product-key separation and the four-split pipeline.
This is implementation validation, not a statistical coverage study. The
interval tests check calculation/behavior, not nominal coverage. See
METHODS_V2.md for propensity-clipping and inference assumptions. The existing
benchmark below concerns the original bounded final-stage regressor; it is not
a full benchmark of the new DR workflow.

A separate full DR fit check (`python benchmark_dr.py --rows 50000`) used seven
numeric features, two randomized arms with known propensities, three folds,
and 80 final trees capped at seven leaves. Recorded one-run results were
1.612 seconds, 146.9 MiB process peak RSS, and 66,486 bytes of final-tree text
serialization. The peak includes imports and data generation. Learned
propensities, many action arms, sparse/dense product features, preprocessing,
selection and calibration can materially change the footprint. This check is
not a full recommendation-pipeline or million-row production benchmark.

Date: October 24, 2026. All fixtures are controlled implementation tests,
separate from the real-data application notebook. No user production data was
available for statistical validation.

## Checks

Run `python -m pytest -q` from the repository root. Recorded result: **24 passed
in 4.36 seconds**. Editable module installation and notebook schema validation
also succeeded. Tests cover:

- Finite-difference verification of the custom gradient and Hessian, including
  exactly zero treatment residuals.
- Algebraic equality of the coordinate objective and full joint residual loss.
- Known binary heterogeneous effects, multiarm coupled effects, and continuous
  slopes, with absolute recovery tolerances in controlled data.
- Reference ordering, action-to-action contrast subtraction, sign reversal,
  identity contrasts, and continuous dose scaling.
- Confounding-only outcome signal: a large naive treatment association is
  removed by the residual procedure in a controlled correctly specified fixture.
- Constant-effect fitting with X omitted or constant X supplied.
- GroupKFold, cross-fit coverage, duplicate/fractional indices, row/group leakage,
  malformed Y/T/W/groups, and failed-refit prediction invalidation.
- Batching, row-wise contrast endpoints, independent-data diagnostic shape
  checks, trusted joblib serialization, and default LightGBM nuisances.
- End-to-end grouped pipeline fitting, validation selection/refit, fitted
  preprocessing with unseen categories, output creation, and percent/fraction
  discount conversion with explicit boundary filtering.

The tests verify finite-sample behavior on these fixtures. They do not establish
confidence-interval coverage, robustness to unobserved confounding, reference
invariance of regularized heterogeneous fits, or consistency of the specific
bounded coordinate tree estimator under arbitrary data-generating processes.

## Controlled final-stage benchmark

Command: `python benchmark.py --rows 50000`. Fresh process per backend, same
50,000-row, eight-feature fixture and scalar residual-loss target. Treatment
residuals are exactly ±.5, so ordinary weighted transformed regression is
numerically safe in this benchmark. This compares final regressor storage and
training resources, not the full custom-estimator pipeline.

| Backend | Trees | Leaves per tree | Training seconds | Process peak RSS, MiB | Reported model bytes | Nodes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Bounded LightGBM | up to 160 | at most 15 | 0.337 | 147.4 | 248,345 | 4,480 |
| Conventional sklearn RF | 160 | data dependent; min leaf 5 | 29.245 | 257.2 | 116,934,480 | 1,624,090 |

Model bytes are LightGBM text serialization size versus RF node/value array
bytes, not identical serialization formats. Peak RSS includes imports and
fixture generation. The RF is not `CausalForestDML`. Hyperparameters deliberately
contrast bounded trees with a small-leaf forest; an equally bounded RF would
also have much smaller storage. These numbers are one run on this environment,
not a universal performance ratio or an accuracy comparison.

The full estimator still allocates encoded features, nuisance fit slices,
native histograms, and O(n) working arrays, with O(nq) temporary residual disk.
A million-row full-pipeline peak-memory benchmark has not been run. Many
treatment arms can still become expensive. Increasing leaves/trees increases
memory; smaller learning rates can demand more trees.

## Recorded environment

Python 3.12, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
scikit-learn 1.8.0, LightGBM 4.7.0, pytest 9.1.1, joblib 1.5.3.
The broader dependency ranges in `requirements.txt` are compatibility targets,
not a tested version matrix.

## Before interpreting real-data results

Check treatment units and action counts first. Review outcome maturity and
attribution for repeated PDP views, pretreatment status of X/W, user separation,
local overlap, and a temporal evaluation design when predicting future users
or periods. For log1p outcomes, use the documented geometric-scale label.
An exploratory best-action contrast is not an off-policy value estimate or
proof of revenue improvement. No statistical significance claims are generated.
