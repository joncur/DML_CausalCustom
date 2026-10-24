"""Research-informed DR recommendation workflow; no fabricated data or product IDs.

Four disjoint group splits: fit 50%, selection 20%, calibration 10%, test 20%.
Only the selected model is refitted on fit+selection. Calibration is optional.
Product IDs are output keys; product attributes/embeddings are effect features.
"""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from causal_dml import LightweightDRLearner, FinalStageConfig
from pipeline import (PipelineConfig, FittedPipeline, prepare_data, _encoder,
                      _clean_features, write_json, _plot)


@dataclass
class RecommendationConfig(PipelineConfig):
    x_cols: list[str] = field(default_factory=lambda: ['data_definition', 'industry'])
    report_cols: list[str] = field(default_factory=lambda: ['data_definition', 'industry'])
    embedding_cols: list[str] = field(default_factory=list)
    product_col: str | None = None
    log1p_outcome: bool = False  # dollar revenue policy values need original-scale Y
    propensity_floor: float = .001
    min_policy_propensity: float = .01
    known_propensities: list[float] | None = None
    calibrate: bool = False
    min_predicted_gain: float = 0.
    # Known deterministic action costs, in outcome units. Already-net revenue
    # must not have the same discount cost deducted again.
    action_costs: dict = field(default_factory=dict)
    cost_multiplier: float = 1.
    final_candidates: list[FinalStageConfig] = field(default_factory=lambda: [
        FinalStageConfig(n_estimators=80, num_leaves=7, max_depth=3, min_child_samples=200),
        FinalStageConfig(n_estimators=120, num_leaves=15, max_depth=4, min_child_samples=300),
    ])


def _fit_dr(train, cfg, categories, final):
    from lightgbm import LGBMRegressor, LGBMClassifier
    xe, we = _encoder(train, cfg.x_cols), _encoder(train, cfg.w_cols)
    x = xe.fit_transform(_clean_features(train, cfg.x_cols)).astype(np.float32)
    w = None if we is None else we.fit_transform(_clean_features(train, cfg.w_cols)).astype(np.float32)
    defaults = dict(n_estimators=120, learning_rate=.05, num_leaves=15,
                    min_child_samples=100, max_bin=63, reg_lambda=2,
                    subsample=.8, subsample_freq=1, colsample_bytree=.8,
                    n_jobs=cfg.n_jobs, verbosity=-1, random_state=cfg.random_state)
    model = LightweightDRLearner(categories=categories, cv=cfg.cv,
        final_config=final, propensity_floor=cfg.propensity_floor,
        known_propensities=cfg.known_propensities, n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        model_y=LGBMRegressor(**(defaults | cfg.nuisance_y_params)),
        model_t=LGBMClassifier(**(defaults | cfg.nuisance_t_params)))
    model.fit(train['_outcome'].to_numpy(), train['_action'].to_numpy(),
              X=x, W=w, groups=train[cfg.group_col].to_numpy())
    return FittedPipeline(cfg, model, xe, we)


def split_groups(d, group_col, seed):
    def hold(frame, frac, s):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=s).split(frame, groups=frame[group_col]))
        return frame.iloc[a].copy(), frame.iloc[b].copy()
    rest, test = hold(d, .2, seed)
    rest, calibration = hold(rest, .125, seed+1)
    train, selection = hold(rest, 2/7, seed+2)
    return dict(train=train, selection=selection, calibration=calibration, test=test)


def validate_splits(splits, n, group_values):
    """Custom row-index splits may implement externally purged temporal designs."""
    expected = {'train', 'selection', 'calibration', 'test'}
    if set(splits) != expected:
        raise ValueError(f'splits must contain {expected}')
    rows, used_groups = set(), set()
    for name, indices in splits.items():
        a = np.asarray(indices)
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or len(a) == 0
                or a.min() < 0 or a.max() >= n or len(set(a)) != len(a)):
            raise ValueError(f'Invalid {name} indices.')
        g = set(group_values[a])
        if rows & set(a) or used_groups & g:
            raise ValueError('Custom split row/group leakage.')
        rows.update(a)
        used_groups.update(g)


def run_recommendations(df, cfg=None, output_dir='recommendation_outputs', *, split_indices=None):
    """Return fitted bundle and independent test report; inputs are actual data.

    split_indices refers to the eligible prepared frame, NOT the original df.
    Supply four externally purged group-disjoint index arrays for temporal use.
    Time separation and outcome-window maturity remain caller-validated.
    """
    cfg = cfg or RecommendationConfig()
    if not cfg.discrete_treatment:
        raise ValueError('Recommendation mode requires categorical interventions. Use CustomCausalDML for scalar dose slopes.')
    if cfg.log1p_outcome and cfg.action_costs:
        raise ValueError('Dollar costs cannot be subtracted from log outcomes.')
    if len(cfg.embedding_cols) > 128:
        raise ValueError('Use a fixed product representation of at most 128 numeric dimensions.')
    for col in cfg.embedding_cols:
        if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError('Embedding columns must be precomputed numeric features.')
    cfg = replace(cfg, x_cols=list(dict.fromkeys(cfg.x_cols+cfg.embedding_cols)),
                  passthrough_cols=list(dict.fromkeys(cfg.passthrough_cols+([cfg.product_col] if cfg.product_col else []))))
    if cfg.product_col and cfg.product_col in cfg.x_cols+cfg.w_cols:
        raise ValueError('product_col is an output key; use product attributes instead of ID features.')
    if not set(cfg.report_cols) <= set(cfg.x_cols):
        raise ValueError('report_cols must be a subset of X features.')
    if not cfg.report_cols or not cfg.final_candidates:
        raise ValueError('Need report columns and at least one candidate.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, action_grid, audit = prepare_data(df, cfg)
    if split_indices is None:
        splits = split_groups(d, cfg.group_col, cfg.random_state)
        audit['split_design'] = 'Random disjoint groups; does not establish future-time generalization.'
    else:
        validate_splits(split_indices, len(d), d[cfg.group_col].to_numpy())
        splits = {name: d.iloc[idx].copy() for name, idx in split_indices.items()}
        audit['split_design'] = 'Caller-provided group-disjoint splits; caller must verify time purging/maturity.'
    counts = splits['train']['_action'].value_counts()
    allowed = [a for a in action_grid if counts.get(a, 0) >= cfg.min_action_rows]
    if cfg.reference not in allowed or len(allowed) < 2:
        raise ValueError('Reference or alternative action has insufficient training support.')
    categories = [cfg.reference]+[a for a in allowed if a != cfg.reference]
    if cfg.known_propensities is not None and categories != action_grid:
        raise ValueError('Known randomization probabilities require categories in exact explicit order with no action filtering.')
    preflight = pd.DataFrame({'action': action_grid})
    for name, frame in splits.items():
        preflight[name+'_rows'] = preflight.action.map(frame['_action'].value_counts()).fillna(0).astype(int)
        splits[name] = frame[frame['_action'].isin(categories)].copy()
    preflight['included'] = preflight.action.isin(categories)
    preflight.to_csv(out/'preflight_actions.csv', index=False)
    audit['splits'] = {name: dict(rows=len(f), groups=int(f[cfg.group_col].nunique())) for name, f in splits.items()}
    if any(len(f) == 0 for f in splits.values()):
        raise ValueError('A split is empty after action filtering.')
    train, select = splits['train'], splits['selection']
    scores, best, best_mse, final = [], None, np.inf, None
    for i, setting in enumerate(cfg.final_candidates):
        candidate = _fit_dr(train, cfg, categories, setting)
        x, w = candidate.encode(select)
        diag = candidate.estimator.dr_diagnostics(select['_outcome'].to_numpy(), select['_action'].to_numpy(),
            X=x, W=w, groups=select[cfg.group_col].to_numpy())
        scores.append(dict(candidate=i, settings=asdict(setting), **diag))
        if diag['dr_pseudo_mse'] < best_mse:
            best, best_mse, final = candidate, diag['dr_pseudo_mse'], setting
    fit_data = pd.concat([train, select]) if cfg.refit_train_validation else train
    if cfg.refit_train_validation:
        best = _fit_dr(fit_data, cfg, categories, final)
    if cfg.calibrate:
        cal = splits['calibration']
        x, w = best.encode(cal)
        best.estimator.fit_calibration(cal['_outcome'].to_numpy(), cal['_action'].to_numpy(),
            X=x, W=w, groups=cal[cfg.group_col].to_numpy())
    costs = np.array([cfg.action_costs.get(a, 0.) for a in categories], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Known costs must be finite and nonnegative.')
    unknown = set(cfg.action_costs)-set(categories)
    if unknown:
        raise ValueError(f'Unknown cost action labels: {unknown}')
    test = splits['test']
    x, w = best.encode(test)
    # Coarse cohort support uses fit rows only, never the test outcomes.
    support = fit_data.groupby(cfg.report_cols+['_action'], dropna=False, observed=True).size().rename('fit_rows').reset_index()
    cohort_keys = test[cfg.report_cols].reset_index(drop=True)
    eligible = np.zeros((len(test), len(categories)), dtype=bool)
    for j, action in enumerate(categories):
        count = cohort_keys.merge(support[support['_action'] == action].drop(columns='_action'),
                                 on=cfg.report_cols, how='left')['fit_rows'].fillna(0).to_numpy()
        eligible[:, j] = count >= cfg.min_cohort_action_rows
    rec = best.estimator.recommend(x, W=w, costs=costs, eligible=eligible,
        min_propensity=cfg.min_policy_propensity, min_gain=cfg.min_predicted_gain,
        cost_multiplier=cfg.cost_multiplier)
    evaluation = best.estimator.evaluate_policy(test['_outcome'].to_numpy(), test['_action'].to_numpy(),
        X=x, W=w, policy=rec['action'], baseline=categories[0], groups=test[cfg.group_col].to_numpy(),
        costs=costs, min_propensity=cfg.min_policy_propensity)
    result = test[cfg.report_cols+([cfg.product_col] if cfg.product_col else [])].reset_index(drop=True)
    result['recommended_action'] = rec['action']
    result['predicted_net_gain'] = rec['predicted_net_gain']
    result['supported'] = rec['supported']
    result.to_csv(out/'recommendations_test_population.csv', index=False)
    # Keep row-policy assignment separate from any causal claims about individual products.
    support.to_csv(out/'cohort_action_support.csv', index=False)
    tables = []
    for action in categories:
        effect = best.estimator.effect(x, T0=categories[0], T1=action)
        rows = cohort_keys.assign(effect=effect)
        tab = rows.groupby(cfg.report_cols, dropna=False, observed=True).effect.agg(['mean', 'size']).reset_index()
        tab = tab.rename(columns={'mean':'mean_effect', 'size':'evaluation_rows'})
        tab['_action'] = action
        j = categories.index(action)
        support_rows = cohort_keys.assign(supported=eligible[:, j] & eligible[:, 0])
        st = support_rows.groupby(cfg.report_cols, dropna=False, observed=True).supported.all().reset_index()
        tab = tab.merge(st, on=cfg.report_cols)
        if cfg.log1p_outcome:
            tab['shifted_geometric_pct'] = 100*np.expm1(tab.mean_effect)
        tables.append(tab)
    cohort = pd.concat(tables, ignore_index=True)
    cohort.to_csv(out/'cohort_effects.csv', index=False)
    _plot(cohort, cfg.report_cols, out, cfg.log1p_outcome)
    scenarios = []
    for scenario in cfg.scenarios:
        mask = np.ones(len(test), dtype=bool)
        for col, values in scenario.get('filters', {}).items():
            if col not in test:
                raise ValueError(f'Unknown scenario column {col}')
            if values:
                mask &= test[col].isin(values).to_numpy()
        for action in categories:
            e = best.estimator.effect(x[mask], T0=categories[0], T1=action) if mask.any() else np.array([])
            scenarios.append(dict(name=scenario['name'], action=action, rows=int(mask.sum()),
                                  mean_effect=float(e.mean()) if len(e) else None))
    pd.DataFrame(scenarios).to_csv(out/'scenario_effects.csv', index=False)
    report = dict(method='cross-fitted shared-nuisance DR learner', outcome_scale='log1p' if cfg.log1p_outcome else 'original',
        selected_config=asdict(final), overlap_training=best.estimator.overlap_, policy_evaluation=evaluation,
        policy_supported_fraction=float(rec['supported'].mean()),
        raw_mean_observed_test_outcome=float(test['_outcome'].mean()),
        note='Policy estimate is versus the reference, not versus the observed stochastic logging policy. '
             'Do not interpret raw observed outcome mean as a policy-adjusted comparison. '
             'No outcome maturity/time purging is inferred from missing timestamps.', data_audit=audit,
        fold_audit=best.estimator.fold_audit_)
    write_json(out/'diagnostics.json', report)
    write_json(out/'validation_selection.json', scores)
    write_json(out/'config.json', asdict(cfg))
    joblib.dump(best, out/'fitted_recommendation_pipeline.joblib', compress=3)
    return best, report
