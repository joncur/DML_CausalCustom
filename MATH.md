# Final-stage mathematics and algorithm

## 1. Estimand and assumptions

Let $Y$ be one outcome, $T$ a treatment, $X$ effect-modifier features, and $W$
additional pretreatment confounders. Define $Z=(X,W)$. Assume consistency,
no interference where relevant, conditional exchangeability given $Z$, and
sufficient treatment overlap. These assumptions cannot be verified by a good
residual score.

For K categorical treatments, explicitly order labels as
$[a_0,a_1,\ldots,a_{K-1}]$ and define the q=K−1 treatment basis

$$
\phi_j(T)=\mathbf{1}\{T=a_j\},\quad j=1,\ldots,q.
$$

The working conditional-mean model is

$$
E[Y\mid Z,T]=g(Z)+\phi(T)^\top\theta(X).
$$

Here $\theta_j(X)$ is the conditional mean contrast of $a_j$ versus $a_0$.
This restriction allows arbitrary baseline effects of W but assumes W does not
modify treatment effects beyond X. If W does modify effects, the resulting
coefficient is generally a treatment-covariance-weighted projection, not an
unweighted CATE averaged over W. Move important effect modifiers into X.

For scalar continuous treatment, $\phi(T)=T$, q=1. The model is linear in
treatment dose, while the slope can vary nonlinearly with X. The package does
not claim arbitrary continuous dose-response identification.

## 2. Cross-fitting and residualization

Define nuisance functions

$$
m(Z)=E[Y\mid Z],\qquad e(Z)=E[\phi(T)\mid Z].
$$

Split training rows into K folds (K here denotes the number of folds, distinct
from the number of treatment labels above). For row i in fold k(i), fit both
nuisances on the other folds and form

$$
\widetilde Y_i=Y_i-\widehat m^{(-k(i))}(Z_i),\qquad
R_i=\phi(T_i)-\widehat e^{(-k(i))}(Z_i).
$$

Categorical e is a vector of class probabilities excluding the reference;
continuous e is a regression prediction. Probability columns are matched to
the explicit label order, not assumed to follow a library's internal ordering.
When groups are supplied, no user/group can occur in both training and holdout
within a fold. Every training row must be held out exactly once. Custom
splitters undergo the same checks.

Both nuisances use X and W. Only X is passed to the final tree learner. The
observed treatment and its residual are not final-stage input features.

## 3. Orthogonal loss and moment equation

Under the working model, residualization yields

$$
\widetilde Y=R^\top\theta_0(X)+\varepsilon,
\qquad E[\varepsilon\mid Z,T]=0.
$$

The target final-stage objective is the R-loss

$$
L(\theta)=\frac{1}{2n}\sum_i
\left(\widetilde Y_i-R_i^\top\theta(X_i)\right)^2.
$$

Its conditional first-order moment is

$$
E\left[R\{\widetilde Y-R^\top\theta(X)\}\mid X\right]=0.
$$

At the true nuisance functions, first-order derivatives of this moment with
respect to perturbations of m or e vanish: terms involving $E[R\mid Z]$ and
$E[\varepsilon\mid Z]$ are zero. This is Neyman orthogonality. Cross-fitting
limits reuse of each row's outcome in its own nuisance prediction. Orthogonality
does not eliminate finite-sample bias, positivity failures, or final-stage
overfitting, and by itself gives no valid confidence intervals for these trees.

The treatment-residual information matrix is

$$
G(x)=E[RR^\top\mid X=x].
$$

It must be sufficiently nonsingular for local identification. A global
$\widehat G=n^{-1}\sum_i R_iR_i^\top$ eigenvalue diagnostic is implemented,
but can conceal weak identification in individual cohorts. The diagnostic
threshold is `identification_tol * max(1, largest_eigenvalue)`.

## 4. Why a model substitution is not enough

A generic regressor of $\widetilde Y$ on X estimates a residual outcome mean,
not $\theta(X)$. Regressing on $[X,R]$ with an unrestricted tree model does not
enforce linearity in R or expose a coherent treatment-slope API.

For one residual treatment r, the familiar weighted regression identity is

$$
(\widetilde Y-r f(X))^2
=r^2\left(\frac{\widetilde Y}{r}-f(X)\right)^2,\qquad r\ne0.
$$

This explains EconML NonParamDML's scalar/binary reweighting approach. The
quotient is mathematically valid, but may be numerically extreme when r is
small. Here the objective is differentiated directly. A row with r=0 provides
zero slope gradient and zero slope information; no clipping, dropping, or
division is needed. The original residual loss remains well defined there.

For multiple arms, independent scalar fits are generally invalid. For one
categorical draw with probabilities p, the conditional covariance of the
nonreference indicators is

$$
\operatorname{Cov}(\phi(T)\mid Z)
=\operatorname{diag}(p)-pp^\top.
$$

Its off-diagonal elements are $-p_jp_k$. Ignoring these terms mixes effects.
The implementation optimizes the joint loss with coupled coordinate updates.

## 5. Joint constant initialization

Accumulate, in row blocks,

$$
\widehat G=\frac1n R^\top R,\qquad
\widehat b=\frac1n R^\top\widetilde Y.
$$

Initialize all coordinates jointly using

$$
\theta^{(0)}=(\widehat G+\lambda_0 I)^{-1}\widehat b.
$$

`ridge` is $\lambda_0$. It stabilizes this initial solve only; it is not the
final-tree L2 penalty and does not create overlap. With X omitted this constant
model is the final estimator. With X supplied, keep this vector as a baseline
and learn coordinate corrections. Predictions are

$$
\widehat\theta_j(x)=\theta^{(0)}_j+f_j(x).
$$

Initially all $f_j=0$. If a coordinate's proposed tree replacement does not
reduce the full unpenalized training residual MSE, retain its old model.

## 6. Coupled coordinate backfitting

Maintain $s_i=\sum_k R_{ik}\widehat\theta_k(X_i)$. For coordinate j, hold the
other coordinate functions fixed and remove their contribution:

$$
u_{ij}=\widetilde Y_i-s_i+R_{ij}\widehat\theta_j(X_i).
$$

Since the new coordinate has the fixed baseline plus a learned correction,
the tree-label array is

$$
a_{ij}=u_{ij}-R_{ij}\theta^{(0)}_j.
$$

Fit a new correction function $f_j$ to

$$
\ell_{ij}(f_j)=\frac12\left(a_{ij}-R_{ij}f_j(X_i)\right)^2.
$$

This equals the joint loss with the other functions fixed, row for row. The
exact derivatives with respect to the raw scalar prediction $f_{ij}$ are

$$
g_{ij}=R_{ij}\left(R_{ij}f_{ij}-a_{ij}\right),\qquad
h_{ij}=R_{ij}^2.
$$

`orthogonal_derivatives()` implements these expressions; finite differences
test both. The gradient and Hessian are not generic outcome regression
derivatives: they include residual treatment and its squared magnitude.

The full per-row vector Hessian is $R_iR_i^\top$, including off-diagonal terms.
LightGBM's scalar custom objective sees its coordinate curvature $R_{ij}^2$;
the repeated partial-residual updates account for coupling across coordinates.
This is approximate functional coordinate descent over bounded tree ensembles,
not a simultaneous full-Hessian Newton solver and not a convergence theorem
for nonconvex tree search.

After fitting, compute the new coordinate prediction and proposed joint fit

$$
s_i^{\rm new}=s_i+R_{ij}
\left(\widehat\theta_j^{\rm new}(X_i)-\widehat\theta_j(X_i)\right).
$$

Accept only if its full-data joint training MSE is finite and strictly lower.
Stop after `backfit_rounds` sweeps or when a sweep's improvement is at most
`tolerance * max(previous_loss, 1e-12)`. A replacement model starts from zero
correction and replaces its predecessor; it does not append another entire
ensemble at every sweep. Training-loss monotonicity prevents accepted updates
from worsening training fit; it does not prove generalization.

Regularized fits need not be reference-invariant in finite samples. Explicit
reference ordering and algebraically consistent contrasts within one model
are guaranteed. Refitting with another reference can produce different
regularization bias. The constant unregularized-limit identity is tested.

## 7. What happens inside a LightGBM leaf?

For a leaf A, the local second-order approximation for an additive update v
is $G_A v + \tfrac12(H_A+\lambda)v^2+\alpha|v|$, where

$$
G_A=\sum_{i\in A}g_{ij},\qquad H_A=\sum_{i\in A}R_{ij}^2.
$$

The regularized Newton leaf value is

$$
v_A=-\frac{\operatorname{soft}(G_A,\alpha)}{H_A+\lambda}.
$$

LightGBM adds the shrunken leaf prediction $\eta v_A$. L2 adds stability when
residual treatment information is small; L1 soft-thresholds the gradient sum.
Minimum Hessian mass is thus a minimum residual-information requirement,
different from a minimum number of rows. Split gains use analogous sums of
gradients and Hessians. Row subsampling computes these on sampled rows.

The gradient implementation uses half squared loss and **sums** of row
derivatives. Dividing all derivatives by n leaves an unregularized minimizer
unchanged but changes the meaning of fixed L1/L2 and minimum-Hessian settings;
this code does not perform that normalization.

## 8. Forest-inspired regularization, without honest-forest claims

| Control | Default | Purpose |
| --- | --- | --- |
| `n_estimators` | 160 | Hard bound on trees per coordinate |
| `num_leaves`, `max_depth` | 15, 5 | Limit partition complexity |
| `min_child_samples` | 200 | Discourage tiny, noisy subgroups |
| `min_child_weight` | 1.0 | Minimum sum of residual squared information |
| `subsample`, bagging frequency | .7, 1 | Random row subsets each tree |
| `colsample_bytree` | .8 | Random feature subsets |
| `extra_trees` | True | Random candidate split thresholds |
| `learning_rate` | .05 | Shrinkage per update |
| `reg_lambda`, `reg_alpha` | 5, 0 | Leaf L2 and L1 penalties |
| `max_bin` | 63 | Smaller histogram storage |

The implementation uses GBDT. Randomization resembles some forest variance
controls, but sequential boosting and this loss remain different from averaging
independent trees. No randomization setting recreates honesty or causal forest
inference. A smaller `min_child_samples` can recover fine structure but raises
variance; a small learning rate with enough trees can still overfit.

## 9. Validation and held-out residual scores

The pipeline reserves train, validation and test users before any fitted
encoding. Each candidate is cross-fitted entirely inside training data.
Its nuisances are then refitted on training data only and used to compute
residuals on the independent validation split. Select the candidate with the
lowest validation residual MSE. Optional final refitting uses train+validation;
test rows are used once, after selection. The default has two candidates,
and they share nuisance hyperparameters, so residual-loss comparisons have
a common nuisance specification.

For a held-out set H,

$$
\widehat L_H=\frac1{|H|}\sum_{i\in H}
\left(Y_i-\widehat m_{\rm train}(Z_i)
-\{\phi(T_i)-\widehat e_{\rm train}(Z_i)\}^\top
\widehat\theta(X_i)\right)^2.
$$

Report $1-L_H/L_{H,\mathrm{constant}}$ against the training constant model.
Do not compare training OOF residual loss with held-out loss as though they
used the same nuisance predictions. A negative held-out R-score indicates
that learned heterogeneity did worse than the training constant on that set.
Neither R-score nor prediction accuracy confirms no hidden confounding.

The global covariance diagnostic and cohort counts are screening tools only.
There are no analytical CIs. If uncertainty is added later, resampling must
respect groups/time, rerun the relevant fitting/selection procedure, and be
validated for the target estimand. Tree-seed dispersion is not a standard error.

## 10. Effects, axes and log outcomes

Categorical contrast:

$$
\widehat\tau(x;a,b)=\widehat\theta_b(x)-\widehat\theta_a(x),
\qquad \widehat\theta_{a_0}(x)=0.
$$

Continuous contrast:

$$
\widehat\tau(x;t_0,t_1)=\widehat\theta(x)(t_1-t_0).
$$

For a supplied population X, `ate` computes the empirical mean of these
contrasts. It is not a separate debiased AIPW ATE estimator. `const_marginal_effect`
always retains the q axis; `effect` returns a one-dimensional vector.

Let $V=\log(1+Y)$ for nonnegative revenue Y. Then

$$
\Delta(x)=E[V(1)\mid X=x]-E[V(0)\mid X=x].
$$

Define the shifted geometric mean $G_t(x)=\exp(E[V(t)\mid X=x])$. Therefore

$$
\frac{G_1(x)}{G_0(x)}-1=\exp(\Delta(x))-1.
$$

This is a ratio of geometric means of Y+1. In general
$E[Y(t)\mid X]\ne\exp(E[V(t)\mid X])-1$. Consequently Δ alone cannot recover
arithmetic expected revenue or its percent lift. Also, exponentiating a mean
Δ differs from averaging `expm1` of row effects. The pipeline uses
`expm1(mean_effect)`, a geometric aggregation on the evaluation population.

For exploratory lead mitigation on that scale, define mean log contrasts
$\delta_L=\tau(\mathrm{lead},0\%)-\tau(0\mathrm{d},0\%)$ and
$\delta_D=\tau(\mathrm{lead},d)-\tau(\mathrm{lead},0\%)$. If $\delta_L<0$,

$$
\mathrm{recovery}=
\frac{\exp(\delta_L)\{\exp(\delta_D)-1\}}
{1-\exp(\delta_L)}.
$$

The numerator uses the same baseline as the lead loss. A ratio is undefined
when there is no lead penalty. It is not an arithmetic-revenue recovery ratio.

## 11. Memory and computational scaling

Let n be rows, p encoded features, q nonreference actions, B trees/coordinate,
and L leaves/tree. Retained final tree storage is O(qBL). The default hard
bound is 160×15 leaves per coordinate (actual models can stop earlier).
This replaces trees whose leaf counts grow with n at low minimum leaf sizes;
an equally bounded conventional forest could also have small model storage.

Temporary treatment residual disk storage is 4nq bytes (float32). Residual
outcomes, current joint prediction, gradients/Hessians and coordinate arrays
are O(n). The moment matrix is O(q²), accumulated without an n×q×q tensor.
OOF prediction probabilities are generated per fold, not retained as a second
full n×q array. Histograms are constructed once and reused across coordinates.
Each coordinate fit replaces a bounded ensemble. Cross-fitting runs nuisance
models serially and discards them; full-data nuisance refits remain for diagnostics.

Fold feature slicing, X/W concatenation, numeric encoding, LightGBM bins and
native working memory still use RAM. Disk memmap can also populate the OS page
cache. The learner is memory-conscious batch training, not fully streaming.
Large action grids still multiply both disk and compute; merge unsupported
bins or reduce the action vocabulary with a defensible intervention definition.

## 12. Provenance and references

An earlier audit archive (`DRPolicy_Work_DML_audit.zip`) was inspected for API
contracts and defects. Its candidate used weighted quotient backfitting. This
version independently implements direct custom-objective backfitting, adds
constant-X handling, batching, validation selection, and a separate workflow.
It does not claim source compatibility with an unavailable earlier standalone
package. The controlled tests exercise the current implementation.

1. [EconML DML formulation](https://www.pywhy.org/EconML/spec/estimation/dml.html).
2. [EconML NonParamDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.NonParamDML.html): scalar/binary reweighting contract.
3. [EconML DML source](https://www.pywhy.org/EconML/_modules/econml/dml/dml.html).
4. [Nie & Wager, Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://arxiv.org/abs/1712.04912).
5. [LightGBM custom objective interface](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html).
6. [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

The coordinate derivation and engineering choices above describe this package;
they are not a claim that EconML or the cited papers implement this exact algorithm.
