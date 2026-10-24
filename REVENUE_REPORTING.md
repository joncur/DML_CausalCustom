# Reporting revenue changes in euros and percentages

## Keep the log model and add a mean revenue view

`log1p(revenue)` remains a useful measurement outcome when zeros and large
orders make revenue difficult to model. Keep that analysis. It estimates a
different target from arithmetic expected revenue; it cannot, by itself,
identify a euro effect. The earlier suggestion that exponentiating any log1p
effect gives revenue percentage lift was too broad.

For the business view, estimate the expected revenue under each action on a
common eligible population. Then report these three quantities:

```text
Baseline expected revenue per opportunity = average predicted/DR mean under reference
Action expected revenue per opportunity   = average predicted/DR mean under action
Euro lift per opportunity                 = action mean - baseline mean
Percentage revenue lift                   = 100 * (action mean / baseline mean - 1)
```

The percentage is undefined if the baseline mean is zero. For a population,
take the ratio of the mean revenues, not the average of individual percentages.
For example, EUR 2.00 versus EUR 2.12 per eligible PDP exposure means EUR 0.12
and 6% lift. Multiplying by 100,000 comparable opportunities gives EUR 12,000
under the same population and exposure assumptions; repeated views cannot
automatically be counted as independent additional sales opportunities.

## First lightweight candidate for the euro model

Use the existing DR recommendation workflow with a **Tweedie log-link outcome
nuisance**. A log link constrains and models the mean multiplicatively while
the label and predictions remain revenue in euros. This is different from
regressing the transformed label `log1p(revenue)`.

```python
from recommendation_pipeline import RecommendationConfig, run_recommendations

cfg = RecommendationConfig(
    outcome_col='revenue',
    log1p_outcome=False,  # the euro-view estimand; your log measurement model is separate
    x_cols=['data_definition', 'industry'],
    nuisance_y_params={
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'n_estimators': 120,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_child_samples': 200,
        'max_bin': 63,
    },
)
fitted, report = run_recommendations(df, cfg)
```

Tune the variance power using training/selection data, for example 1.2, 1.5,
and 1.8; no value is universally best. For powers strictly between one and
two, Tweedie modeling accommodates nonnegative continuous values and a mass
at zero. It requires nonnegative revenue, so use another model if returns
produce negative net outcomes. LightGBM documents Tweedie as a log-link
objective [1]. The alternative hurdle design below separates the two processes.

**Apply Tweedie to the outcome nuisance only.** DR pseudo-outcomes, residuals,
and causal contrasts can be negative, so their final regression must retain
the signed squared-error objective. The code already preserves that separation.
This configuration is supported by the current pipeline; its accuracy on your
data and the best variance power have not been established.

## Two part revenue modeling

Another candidate factors the conditional mean exactly:

```text
Expected revenue under action a
  = P(revenue > 0 | action a, features)
    * E(revenue | revenue > 0, action a, features)
```

Fit purchase probability with a classifier. Fit positive purchase amounts
with a log-link Gamma/Tweedie regressor, or with a log-amount model plus a
proper conditional retransformation correction. Both predictions feed the
same original-revenue DR score. This permits interventions to change both
purchase probability and order value. It adds a second nuisance model and
needs enough purchasers per relevant region of the data. It is described
here as an alternative, not implemented by the one-regressor configuration.

For this approach, log of strictly positive amount is often more natural
than log1p because the purchase classifier already handles zeros. A log-link
mean model avoids that retransformation problem.

## If the outcome nuisance must remain log1p

Let `m_a(z)` be the conditional mean of log1p revenue for action a. An exact
conditional retransformation identity is:

```text
S_a(z) = E[exp(log1p(Y) - m_a(z)) | action a, z]
Mean revenue_a(z) = exp(m_a(z)) * S_a(z) - 1
```

This is the smearing-factor route. The correction may differ by action and
covariates; one global residual correction is not generally sufficient under
heteroskedasticity. Estimate and validate it within the training design, using
honest residual predictions and an independent check on original revenue.
Do not exponentiate the CATE alone: both action-level outcome predictions and
the retransformation factors are needed. [2]

`100 * expm1(log_CATE)` remains a legitimate shifted geometric-mean contrast,
labelled as such. It is not a generally valid revenue percentage. Log-like
transforms with zero outcomes also depend on the outcome's units; keep euros
consistent instead of mixing euros and cents. [3]

## What no transformation can eliminate

If rare large orders account for a large share of actual revenue, they also
account for a large share of uncertainty about its arithmetic mean. A log
link, a hurdle model or smearing can improve modeling but cannot eliminate
that statistical fact. Retain independent user-group evaluation, monitor
overlap/importance-weight ESS, and show uncertainty on euro contrasts.
Winsorizing changes the outcome to capped revenue; report that separately if
used rather than calling it total revenue.

The practical recommendation is two views: keep the current log1p analysis for
stable effect exploration, and validate a separate mean-revenue DR view for
euro and percentage decisions. Compare Tweedie against the existing L2 nuisance
on the same untouched evaluation design before choosing a production model.

## References

1. LightGBM parameters, Tweedie and Gamma log-link objectives. https://lightgbm.readthedocs.io/en/stable/Parameters.html
2. Manning and Mullahy, Estimating log models to transform or not to transform. Research on skewed outcomes and heteroskedastic retransformation. https://www.nber.org/system/files/working_papers/t0246/t0246.pdf
3. Chen and Roth, Logs with Zeros Some Problems and Solutions, 2024. https://ungated-research.s3.amazonaws.com/pdf/3604-QJE-2024-2-Logs_with_Zeros__Some_Problems_and_Solutions.pdf
4. GitHub mathematical-expression documentation. https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
