# Lightweight causal learning for ecommerce decisions

Research review and implementation rationale

Reviewed October 24 2026

## Main recommendation

Use a small orthogonal model family with separate measurement and decision
evaluation. Retain the residual-loss R-learner for categorical measurements
and scalar continuous slopes. Add a cross-fitted doubly robust learner for
categorical CATEs and recommendations, and evaluate the resulting fixed policy
on untouched users. Both final stages use bounded LightGBM ensembles. There
is no universally best DML estimator; selection and identification matter more
than adding model complexity. [R1-R5,R8]

For the current project, use `data_definition` and `industry` as effect
features, with other genuine pretreatment variables in W. Keep product IDs as
join keys. The treatment is an intervention such as carousel exposure or a
lead and discount action, not the identity of one of 500k products. Add stable
catalog attributes before expanding to fixed-size frozen product embeddings.
This design permits product-specific predictions through shared features,
without claiming that rare products have individually identified effects.

The most consequential upgrade is independent policy evaluation on original
revenue or contribution margin. Maximizing fitted lift is not evidence that
the chosen policy improves business value. For `log1p(revenue)`, exponentiating
an effect gives a geometric ratio for revenue plus one; it does not recover
arithmetic expected revenue. The recommendation workflow therefore defaults
to the original outcome scale. Heavy tails remain a real estimation problem.

## Review scope and currency

The review covers foundational orthogonal estimation, residual and DR CATE
learning, current EconML and DoubleML documentation and releases, calibration,
policy evaluation, and primary-source ecommerce applications. The references
include 2025 and 2026 work. This is a focused implementation review rather
than an exhaustive systematic review of every DML paper.

The current EconML release page lists 0.17.0, including DRLearner sample
trimming and updated dependency support. Several rendered API pages still
show 0.16.0, so the release notes and API page versions must be distinguished.
DoubleML's current release notes list 0.11.4 and recent additions covering
repeated CATE fitting, static panel effects, and tuning. These developments
support more explicit validation and estimand choices; they do not imply that
all methods should be combined into one estimator. [R9,R10]

The review also checked withdrawal status. The current record for Entire
Chain Uplift Modeling marks it withdrawn after a January 2026 revision. Its
earlier deployment claim is not used here as affirmative adoption evidence.
UniMVT is an available September 2026 preprint accepted for CIKM 2026; the
conference is scheduled after this review date. Its deployment statements
are attributed to the authors, not independently audited. [R17,R22]

## Methods that materially affect the implementation

### Orthogonality and treatment semantics

DML combines nuisance prediction with orthogonal scores and cross-fitting.
It does not turn an arbitrary outcome predictor into a causal model. A causal
forest solves local moment equations; swapping in a generic regressor changes
the estimator. The retained R-learner differentiates the residual objective
directly and uses coupled updates for multiple categorical treatment residuals.
The mathematical details are in MATH.md. [R1,R2,R4]

The new DR learner instead estimates an outcome conditional on treatment and
covariates, plus treatment propensities. It constructs cross-fitted AIPW
potential-outcome scores and regresses their contrasts on X. Separate final
arm regressions are valid in this construction because the pseudo-outcomes
already encode the appropriate causal contrast. This is not independent
regression of each treatment residual against the same outcome residual. [R3,R5]

The shared outcome S-learner reduces nuisance model count. Its weakness is
possible underfitting of treatment interactions. Correct propensities can
protect the DR score under the usual assumptions, but both nuisance models
still require diagnostic attention. Final squared-error regression retains
a conditional mean target; silently replacing it with a robust or quantile
loss would change that target.

### Overlap and inference

Recent trimming support reinforces the importance of stating which population
is estimated. The implementation records propensity clipping and recommendation
support separately. It does not silently drop unsupported evaluation rows and
continue to call the result a population effect. Clipping introduces a
bias-variance tradeoff and weakens the usual correct-propensity robustness
claim below the floor. [R5,R9]

The pipeline separates training, candidate selection, optional calibration,
and final evaluation by user. Clustered fixed-policy Wald intervals use test
scores; they are not forest CATE intervals or finite-sample guarantees.
Repeated nuisance fitting and formal sensitivity analysis are valuable
extensions, documented in DoubleML, but are not claimed as implemented here.
Hidden confounding, outcome attribution and repeated-exposure interference
remain identification issues. [R8,R11,R24]

### Calibration and current research directions

Causal isotonic calibration and 2025 orthogonal calibration research motivate
an optional monotone map from predicted CATE to independent DR scores. This
adds little model storage. The package implements this restricted design,
not the full general calibration framework or its guarantees. It stays off
by default because sparse, noisy calibration sets can worsen decisions. [R6,R7]

The 2025 pretraining work suggests borrowing useful prognostic structure for
heterogeneity estimation. For a product catalog, fixed semantic attributes
are a plausible low-cost starting point. Any representation learned from
outcomes must respect training boundaries. The paper's domain-specific
results do not establish that product embeddings improve this dataset. [R19]

Many-treatment fusion and October 2026 distributionally robust policy learning
address sparse actions and deployment shift. They are research options, not
features of this implementation. Treatment fusion can change the business
meaning of an intervention; a production fusion rule needs independent
validation. The October preprint is especially recent and should not be
treated as mature deployment evidence. [R20,R21]

## Documented organizational applications

| Organization | Publicly documented use | Evidence status | Method distinction |
| --- | --- | --- | --- |
| Amazon | Estimating effects of customer actions at large scale | 2023 company research publication | Explicit DML; distributed Spark system, not this lightweight implementation [R13] |
| Amazon FBA | Effects of seller adoption of inventory recommendations | 2024 company technical account; estimates surfaced to sellers | Explicit DML and selection-bias adjustment [R14] |
| Booking.com | Personalization and financially viable promotional campaigns | Practitioner-authored 2023 tutorial based on internal training and application experience | Uplift and constrained promotions; not proof of this exact DML model [R15] |
| ZOZO and ZOZOTOWN | Fashion recommendation and logged-policy evaluation | Company dataset page describing randomized logging and propensities | Bandit and OPE infrastructure; not proof of DML deployment [R16] |
| Kuaishou | Coupon uplift and debiased base CTR in live-stream ads | UniMVT authors report A/B testing and subsequent deployment in 2026 | Neural multi-valued causal model, not lightweight DML [R17] |
| Uber | Uplift estimation tooling | Company-maintained CausalML repository | Public causal/uplift toolkit; repository alone does not establish a specific campaign deployment [R23] |

These examples establish that the decision problems are used in practice.
They do not validate a causal effect or expected ROI for this catalog.
Reported gains from another platform should not be used as planning estimates.

## Prioritized target applications for this ecommerce project

### Carousel and recommendation module exposure

Prioritize this now. Treatment should be eligibility or assigned exposure to
the carousel versus no carousel, defined before downstream engagement. Use
attributed purchase, original-scale revenue, or contribution margin as the
outcome, with data_definition and industry in X. Avoid treating a click on
the carousel as though it were a randomized display assignment. Log the
decision opportunity, exposure, available actions, and any randomization
probability. ZOZO supplies an adjacent recommendation-evaluation example;
it does not establish that observational PDP exposure is unconfounded. [R16]

### Joint discount and delivery mitigation

Prioritize this now, with the existing action bins. Compare supported joint
actions and fixed-lead discount contrasts. A shipping promise may be an
intervention only if the business can actually change it. Stock availability,
inventory and customer urgency can confound delivery effects. Minimize lost
contribution, not merely maximize conversion. Booking.com's promotion work
supports the decision class; the joint industrial-catalog application is a
proposed extension. [R15]

### Suppression of unnecessary discounts

Allocate offers where incremental net value is positive; retain a genuine
no-offer reference and a randomized holdout where feasible. High purchase
propensity is not evidence of high treatment responsiveness. Coupons with
costs conditional on redemption need an incremental cost model, not a fixed
per-view subtraction. Booking.com and Kuaishou document related promotional
applications using different model architectures. [R15,R17]

### Product information enrichment

Test enhanced specifications, application guidance, compatibility details,
or a technical comparison block. The treatment is the information change,
not the product's category. Freeze pretreatment catalog representations to
avoid incorporating content generated after the treatment. This is a high-fit
proposed application for a technical catalog; the references above do not
establish a named company's use of this exact design.

### Substitution and cross sell quality

Compare carousel strategies such as compatible substitute versus complementary
item versus no module, using total basket or account revenue rather than only
revenue of the promoted SKU. Cannibalization, displaced orders and interference
between displayed items are central. A single-row treatment model cannot
separate all slate interactions. Begin with a small randomized strategy set;
move to slate or network methods only with the necessary exposure logs.
This is an extension of recommendation OPE, not a claim of independent
identification for 500k product arms. [R16]

### Customer contact and channel allocation

Compare eligible email, sales contact, onsite offer and no contact within a
defined opportunity window. Channel can be treatment or confounder depending
on the decision; it cannot be assigned both roles mechanically. Sales teams
often contact users with private purchase intent, so observational results
need a strong confounder audit and an experimental check. Multi-action uplift
tooling is relevant, but this specific B2B application is proposed. [R23]

### Inventory and availability interventions

Measure back-in-stock notices, substitute suggestions, or operational
recommendations. Amazon FBA is a direct organizational analogue for the
effect of following recommendations, not evidence that delivery lead alone
is randomly assigned. Inventory decisions can influence other products and
future customers; define the interference and outcome horizon explicitly. [R14]

### Retention and reactivation

Choose contact or incentive among lapsed users based on incremental return,
with recency measured before the decision. Separate immediate orders from
purchase timing shifts and long-term retention. Repeated treatments require
a longitudinal design if prior actions affect later confounders; the current
one-step estimator should not be relabeled DynamicDML. This is a proposed
extension of personalized promotion targeting. [R15]

### Search layout and assisted purchasing

Evaluate guided search, technical filters, comparison tools, quotation
assistance or a conversational assistant as small intervention categories.
Use downstream purchase and contribution outcomes, not only clicks. The
Kuaishou work illustrates that intervention-induced clicks can contaminate
base CTR estimation, although it concerns advertising rather than technical
catalog search. Start with exposure assignment experiments where possible. [R17]

### Shipping incentives and commercial terms

Evaluate free shipping, order-threshold offers, or standardized quote follow-up
actions under eligible commercial constraints. Costs, quote negotiation,
account agreements and latent urgency must be logged. This is a proposed
application; no reference here establishes company adoption of this exact
industrial-sales design. Continuous prices or contract terms need a separate
dose-response or demand model beyond a constant-in-dose slope.

## Scaling from coarse definitions to 500k products

Use three stages. First, establish reliable coarse estimates for data_definition
and industry with the existing treatments. Second, add measured catalog
attributes such as application, material class, purity, pack size, compatibility
and stock status where they are pretreatment. Keep geography in X if it changes
effects relevant to decisions; otherwise include it in W as appropriate.

Third, add fixed-size catalog representations and test whether they improve
held-out causal scores and policy value. UMLC studies user-context interactions
and context grouping; it motivates investigating context compression, not
assuming that its neural modules or clustering should be copied into a small
LightGBM pipeline. The default implementation accepts up to 128 numeric
embedding columns and reports on coarse cohorts independently of X. [R18]

A 500k by 64 float32 product table is about 122 MiB. Replicating the vectors
over four million exposure rows is about 977 MiB before encoders, slices and
histograms. Product-level sparsity remains a statistical problem even if
feature storage is small. Predicting the effect for a rare product through
shared attributes is not the same as estimating a standalone product effect.
Monitor coverage by taxonomy depth, frequency, and product age; reserve a
product-disjoint holdout when cold-start products are the actual target.

## Delivered pipeline and implementation boundaries

The reusable module contains both estimators with stable treatment contrast
axes. The new recommendation pipeline uses original-scale outcomes, a shared
outcome nuisance, bounded final regressors, group-separated selection and
evaluation, optional independent isotonic calibration, support checks, known
action costs, and a fixed-reference policy comparison. Configuration and a
notebook remain outside the estimator. The earlier pipeline is retained.

The final tree storage grows with action count times bounded trees times
bounded leaves, rather than with an unrestricted number of leaves per tree.
Float32 DR contrasts use temporary memmap storage. A single arm target is
loaded for each final regression. Fold nuisances are discarded. Dense feature
matrices, fold slices and native LightGBM working storage still use memory;
the implementation is not streaming or distributed training.

There is no automatic representation learning, general dose-response model,
sequential policy estimator, treatment fusion, interference adjustment, formal
sensitivity bound, or exact global budget optimizer. Costs use a configurable
shadow multiplier rather than a hard knapsack constraint. These are deliberate
scope boundaries to keep the implementation testable and small. The math and
assumptions are documented in METHODS_V2.md; test and benchmark limits are in
VALIDATION.md. Production performance remains to be measured on actual data.

## References and how they were used

All links below were reviewed on October 24 2026. Company accounts and
author-reported deployments are not independent replications.

- **R1** Chernozhukov and colleagues. Double Debiased Machine Learning for Treatment and Causal Parameters. Foundational paper; arXiv record revised 2024. https://arxiv.org/abs/1608.00060 . Used for orthogonality and cross-fitting rationale.
- **R2** Nie and Wager. Quasi Oracle Estimation of Heterogeneous Treatment Effects. 2017 preprint, revised 2020. https://arxiv.org/abs/1712.04912 . Used for residual-loss CATE formulation.
- **R3** Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects. 2020 preprint, revised 2023. https://arxiv.org/abs/2004.14497 . Used for DR pseudo-outcome learning and limits of theoretical transfer.
- **R4** EconML. Orthogonal Double Machine Learning guide and CausalForestDML API. https://www.pywhy.org/EconML/spec/estimation/dml.html and https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html . Used to distinguish residual learning from forest local moment estimation.
- **R5** EconML. Doubly Robust Learning guide and DRLearner API. https://www.pywhy.org/EconML/spec/estimation/dr.html and https://www.pywhy.org/EconML/_autosummary/econml.dr.DRLearner.html . Used for treatment-conditional outcome nuisances, contrasts, and propensity clipping.
- **R6** Van der Laan, Ulloa Perez, Carone and Luedtke. Causal Isotonic Calibration for Heterogeneous Treatment Effects. ICML 2023. https://proceedings.mlr.press/v202/van-der-laan23a.html . Motivates optional independent isotonic calibration.
- **R7** Whitehouse, Jung, Syrgkanis, Wilder and Wu. Orthogonal Causal Calibration. COLT 2025 extended abstract. https://proceedings.mlr.press/v291/whitehouse25a.html . Motivates causal-score rather than ordinary predictive calibration; full framework not reproduced.
- **R8** Athey and Wager. Policy Learning with Observational Data. Preprint revised 2020. https://arxiv.org/abs/1702.02896 . Used for the distinction between effect fitting and constrained policy learning/value evaluation.
- **R9** EconML. Release 0.17.0 notes. https://github.com/py-why/EconML/releases/tag/v0.17.0 . Current trimming, compatibility and bug-fix review; API pages can lag.
- **R10** DoubleML. Current release notes. https://docs.doubleml.org/stable/release/release.html . Reviewed through listed version 0.11.4, including recent CATE and panel changes.
- **R11** DoubleML. Sensitivity analysis guide. https://docs.doubleml.org/stable/guide/sensitivity.html . Identifies formal omitted-variable sensitivity as a separate extension, not an implemented feature.
- **R12** LightGBM. Parameters and Python train API. https://lightgbm.readthedocs.io/en/stable/Parameters.html and https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html . Defines tree bounds, sampling, regularization and custom gradient/Hessian interface.
- **R13** More, Kotwal, Chappidi, Mandalapu and Khawand. Double machine learning at scale to predict causal impact of customer actions. ECML PKDD 2023, Amazon Science. https://www.amazon.science/publications/double-machine-learning-at-scale-to-predict-causal-impact-of-customer-actions . Explicit ecommerce DML scale example; no transfer of performance claims.
- **R14** Ozer and Simsek. Removing selection bias from evaluation of recommendations. Amazon Science, October 21 2024. https://www.amazon.science/blog/removing-selection-bias-from-evaluation-of-recommendations . Explicit FBA seller recommendation application.
- **R15** Moraes, Proenca, Kornilova, Albert and Goldenberg. Uplift Modeling from Causal Inference to Personalization. CIKM 2023 tutorial. https://arxiv.org/html/2308.09066v1 . Booking.com personalization and constrained promotion application evidence.
- **R16** ZOZO Research. Data Release and Open Bandit Dataset. https://research.zozo.com/data.html . Logged recommendation propensities and OPE use; not claimed as a DML deployment.
- **R17** Yang and colleagues. Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing A Unified Causal Framework. September 8 2026 revision; accepted CIKM 2026. https://arxiv.org/html/2602.12972v3 . Author-reported Kuaishou ad/coupon deployment; neural model not reimplemented.
- **R18** Sun and colleagues. Robust Uplift Modeling with Large Scale Contexts for Real time Marketing. KDD 2025. https://arxiv.org/html/2502.15697v1 . Context grouping and user-item feature motivation, not proof of deployment of this package.
- **R19** Schuessler, Sverdrup and Tibshirani. Statistical Learning for Heterogeneous Treatment Effects Pretraining Prognosis and Prediction. 2025 preprint. https://arxiv.org/abs/2505.00310 . Product representation research direction; not implemented as supervised pretraining.
- **R20** Zhu, Chu, Lipkovich, Ye and Yang. Doubly Robust Fusion of Many Treatments for Policy Learning. May 2025 preprint. https://arxiv.org/abs/2505.08092 . Research option for sparse treatment vocabularies; not implemented or established ecommerce deployment.
- **R21** Mandal. Distributionally Robust Policy Learning with Observational Data. October 18 2026 preprint. https://arxiv.org/abs/2610.25591 . Recent robustness research; not independently replicated here and not implemented.
- **R22** Huang and colleagues. Entire Chain Uplift Modeling with Context Enhanced Learning for Intelligent Marketing. https://arxiv.org/abs/2402.03379 . Current record marks withdrawal; excluded as affirmative method/adoption evidence.
- **R23** Uber. CausalML company-maintained repository. https://github.com/uber/causalml . Public uplift tooling evidence; not a campaign case study.
- **R24** DoubleML. Sample splitting cross fitting and repeated cross fitting. https://docs.doubleml.org/stable/guide/resampling.html . Split design and optional future repeated-fit validation.
