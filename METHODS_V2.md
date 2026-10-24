# Lightweight measurement and recommendation methods

This extension retains `CustomCausalDML` and adds `LightweightDRLearner` in the
same standalone `causal_dml.py` file. There is no universally most accurate DML
model. This implementation uses a small model family, independent selection,
explicit overlap diagnostics, and bounded tree storage.

## Which estimator to use

| Objective | Estimator | Important restriction |
| --- | --- | --- |
| Scalar continuous treatment slope | `CustomCausalDML(discrete_treatment=False)` | Conditional response linear in dose |
| Categorical residual-loss measurement | `CustomCausalDML` | Final X must capture effect modification; see MATH.md |
| Categorical CATE and action recommendation | `LightweightDRLearner` | Conditional exchangeability and positivity; finite action set |
| Fixed policy evaluation on independent rows | `LightweightDRLearner.evaluate_policy` | No selection using the evaluation outcomes; inference assumptions apply |

R-learning and DR-learning are related orthogonal methods but have different
nuisance functions. The DR outcome nuisance includes the observed treatment;
the R outcome nuisance does not. DR contrasts can be fitted separately **after**
the coupled potential-outcome scores have been constructed. This does not
contradict the requirement for coupled coordinate fitting of the R-loss.

## DR scores and final stage

For action a, define conditional outcome and propensity models

$$
\mu_a(Z)=E[Y\mid T=a,Z],\qquad p_a(Z)=P(T=a\mid Z),\quad Z=(X,W).
$$

With nuisance estimates from folds excluding row i,

$$
\Gamma_{ia}=\widehat\mu_a^{(-k(i))}(Z_i)
+\frac{\mathbf1\{T_i=a\}}{\widehat p_a^{(-k(i))}(Z_i)}
\left[Y_i-\widehat\mu_a^{(-k(i))}(Z_i)\right].
$$

The final target is $D_{ia}=\Gamma_{ia}-\Gamma_{i0}$, and

$$
\widehat\tau_a=\arg\min_{f\in\mathcal F}
\sum_i(D_{ia}-f(X_i))^2+\mathrm{tree\ regularization}.
$$

Without clipping, and under the causal identification assumptions, these scores
identify $E[Y(a)-Y(0)\mid X]$ if the corresponding conditional outcome models
are correct **or** the propensity models are correct. They also require a
capable final regression learner to estimate this conditional mean. Unlike the
partially linear R specification, DR scores can average over treatment effect
modification by W. Include a modifier in X if the policy should personalize
to that modifier.

A single shared outcome regressor on `[X,W,one_hot(T)]` estimates all potential
outcomes. This keeps one outcome nuisance per fold instead of one per arm.
It can underfit treatment interactions; the validation suite and independent
scores must be used to assess performance. Double robustness is not a reason
to neglect either nuisance model. There is no retained forest.

The implementation uses `max(p_a, propensity_floor)` in denominators for
numerical stability and records the clipped fraction per arm. Clipping changes
the score: it preserves the correct-outcome branch, but can bias the
correct-propensity branch where true propensities fall below the floor.
Recommendation support uses a separately configured, higher threshold. It
checks both proposed and reference actions; low support is reported rather
than silently redefining the target population by trimming. Experimental
constant known probabilities can be supplied in explicit action order.

Final models use bounded squared-error GBDT with the same sampling, leaf and
depth controls as the R implementation. `backfit_rounds` and `tolerance` are
R-only settings and have no effect on DR fits. DR needs no coupled coordinate
iterations after score construction. Robust/quantile losses would generally
change the final conditional-mean target and are not substituted automatically.

## Calibration

Optional `fit_calibration` fits one monotone isotonic map per nonreference arm
from raw CATE predictions to DR contrasts on a dedicated independent set.
This is a lightweight causal-score calibration design motivated by causal
isotonic calibration and orthogonal calibration research [R6,R7]. It does not
implement every algorithm or theoretical guarantee in those papers.

The estimator refuses to evaluate on fitting or calibration groups when group
IDs are available. Calibration can hurt with weak signal or sparse data; it is
off by default. It does not enforce agreement across arbitrary action-reference
refits or calibrate pointwise uncertainty intervals. All contrasts within one
fitted model remain algebraically consistent.

## Policy and independent value evaluation

Given known deterministic action costs c in outcome units, the plug-in policy
maximizes predicted CATE minus `cost_multiplier * incremental_cost`, subject to
propensity, reference and cohort count support. It falls back to reference when
no eligible predicted improvement exceeds `min_predicted_gain`. A fallback
itself can lack reference overlap; `supported=False` makes this visible.
The multiplier can represent a shadow price, but no exact global knapsack or
hard budget guarantee is implemented. Discount expense already included in
net revenue must not be deducted twice. Redemption-dependent costs need their
own outcome/cost model; they are not deterministic per-impression costs.

On an independent evaluation population, evaluate a fixed policy pi against
a fixed baseline b using

$$
\widehat\Delta_V=\frac1n\sum_i\left[
\Gamma_{i,\pi(Z_i)}-\Gamma_{i,b(Z_i)}
-c_{i,\pi(Z_i)}+c_{i,b(Z_i)}\right].
$$

This is a direct value comparison, not the mean of the largest fitted CATEs.
Costs used for evaluation are actual known costs; the decision multiplier is
not subtracted as an economic expense. The default baseline is the first
treatment category. The comparison is not against the stochastic historical
logging policy. Unsupported-row and clipping fractions are reported for the
full evaluation population, along with importance-weight effective sample size
$\mathrm{ESS}=(\sum_i w_i)^2/\sum_i w_i^2$.

For centered evaluation scores $d_i=\psi_i-\bar\psi$ and G independent user
clusters, the reported cluster sandwich standard error is

$$
\widehat{\mathrm{SE}}=
\frac1n\sqrt{\frac{G}{G-1}\sum_{g=1}^{G}
\left(\sum_{i\in g}d_i\right)^2}.
$$

A Student t critical value with G−1 degrees of freedom gives an approximate
Wald interval. It is conditional on the separately learned policy and
nuisances and requires suitable nuisance convergence, moments, independent
clusters and overlap. Heavy-tailed revenue, few clusters, clipping, hidden
confounding or a policy selected using these evaluation outcomes can invalidate
it. It is not a finite-sample certificate, CATE interval, or simultaneous
confidence band. Multiple policies/scenarios require a separate multiplicity
plan. No test-outcome policy selection is performed by the pipeline.

## Product features and scale

Current defaults use X=`[data_definition, industry]`. Optional `report_cols`
stay coarse even after richer features enter X. `product_id` is an output key,
not a one-hot feature and not a treatment. Optional precomputed numeric
`embedding_cols` add at most 128 dimensions; the encoder can be frozen outside
this package. Use pretreatment catalog content and version its taxonomy and
embedding model. Any outcome-supervised product representation must be trained
inside the relevant splits; the package does not train one for you.

At 500k products, a 64-dimensional float32 lookup table is 128,000,000 bytes,
about 122 MiB. Joining it onto millions of exposure rows can cost much more;
four million dense 64-dimensional rows alone are about 977 MiB. No claim of
out-of-core training is made. Prefer compact physicochemical/catalog attributes,
coarse taxonomy, sparse features and bounded tree counts before adding dense
embeddings. Product-specific predictions borrow structure; rare products do
not suddenly acquire independently identified causal effects.

## Workflow and boundaries

`run_recommendations` uses 50/20/10/20 disjoint groups for train, selection,
calibration and test. It selects a bounded DR setting by independent
pseudo-outcome MSE, optionally refits on train+selection, optionally calibrates,
then reports a single reference-policy comparison on the untouched test set.
The calibration block remains reserved even when calibration is disabled.
Custom group-disjoint split indices can implement an externally purged time
design. They refer to the eligible prepared frame, not original input indices.
No timestamp, outcome maturity or temporal purging is invented.

This package does not implement sequential treatment regimes, general nonlinear
continuous dose responses, instrumental variables, difference-in-differences,
automatic sensitivity bounds, marketplace interference, treatment fusion, or
distributionally robust optimization. Those methods answer additional
questions with additional assumptions. Research references R1–R21 and evidence
statuses are in [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md).
