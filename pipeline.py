"""Configurable real-data workflow. Domain choices stay outside causal_dml.py."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from causal_dml import CustomCausalDML, FinalStageConfig


@dataclass
class PipelineConfig:
    outcome_col: str = 'revenue'
    group_col: str = 'user_id'
    x_cols: list[str] = field(default_factory=lambda: ['industry', 'data_definition', 'region'])
    w_cols: list[str] = field(default_factory=lambda: ['day', 'week', 'month', 'device', 'channel'])
    passthrough_cols: list[str] = field(default_factory=list)
    # Set treatment_col to use an existing categorical/binary/scalar treatment.
    # None builds a joint lead|discount categorical action.
    treatment_col: str | None = None
    discrete_treatment: bool = True
    reference: str | float = '0d|0%'
    categories: list | None = None
    log1p_outcome: bool = True
    discount_col: str = 'discount_percentage'
    lead_col: str = 'lead_time'
    discount_input_is_percent: bool = True  # 7.7 becomes .077; .077 stays .077 when False
    discount_edges: tuple = (0., .01, .03, .06, .10, .20, .45)
    max_discount: float = .45
    max_lead: float | None = None
    min_action_rows: int = 100
    min_cohort_action_rows: int = 100
    sample_group_fraction: float = 1.0
    cv: int = 3
    n_jobs: int = 2
    nuisance_y_params: dict = field(default_factory=dict)
    nuisance_t_params: dict = field(default_factory=dict)
    random_state: int = 2026
    refit_train_validation: bool = True
    final_candidates: list[FinalStageConfig] = field(default_factory=lambda: [
        FinalStageConfig(n_estimators=80, num_leaves=7, max_depth=3, min_child_samples=300),
        FinalStageConfig(),
    ])
    scenarios: list[dict] = field(default_factory=lambda: [
        {'name': 'all', 'filters': {}},
        {'name': 'lab_water', 'filters': {'data_definition': ['lab_water'], 'region': [], 'industry': []}},
    ])


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + '\n')


def prepare_data(df, cfg):
    """Returns eligible copy + explicit filtering audit; never clamps a treatment."""
    required = [cfg.outcome_col, cfg.group_col] + cfg.x_cols + cfg.w_cols
    required += cfg.passthrough_cols
    required += [cfg.treatment_col] if cfg.treatment_col else [cfg.discount_col, cfg.lead_col]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if set(cfg.x_cols) & set(cfg.w_cols):
        raise ValueError('X and W columns must not overlap.')
    excluded = {cfg.outcome_col, cfg.group_col, cfg.treatment_col}
    if cfg.treatment_col is None:
        excluded |= {cfg.discount_col, cfg.lead_col}
    if excluded & set(cfg.x_cols + cfg.w_cols):
        raise ValueError('Outcome, treatment, and group IDs cannot be features.')
    if not cfg.x_cols:
        raise ValueError('Pipeline requires X columns; estimator itself supports X=None.')
    if cfg.min_action_rows < 1 or cfg.min_cohort_action_rows < 1:
        raise ValueError('Support thresholds must be positive.')
    d = df[list(dict.fromkeys(required))].copy()
    audit = {'input_rows': len(d)}
    if d[cfg.group_col].isna().any():
        raise ValueError('Missing group IDs must be resolved before splitting.')
    y = pd.to_numeric(d[cfg.outcome_col], errors='raise').astype(float)
    if not np.isfinite(y).all() or (cfg.log1p_outcome and (y < 0).any()):
        raise ValueError('Outcome must be finite; log1p revenue requires nonnegative values.')
    d['_outcome'] = np.log1p(y) if cfg.log1p_outcome else y
    if cfg.treatment_col is None:
        if not cfg.discrete_treatment:
            raise ValueError('Joint bins require discrete_treatment=True.')
        edges = np.asarray(cfg.discount_edges, dtype=float)
        if (not np.isfinite(edges).all() or len(edges) < 2 or edges[0] != 0
                or not np.all(np.diff(edges) > 0) or edges[-1] > 1
                or not 0 < cfg.max_discount <= edges[-1]):
            raise ValueError('Discount edges/max_discount must be increasing fractions in [0,1].')
        discount = pd.to_numeric(d[cfg.discount_col], errors='raise').astype(float)
        lead = pd.to_numeric(d[cfg.lead_col], errors='raise').astype(float)
        if cfg.discount_input_is_percent:
            discount = discount / 100.
        if (not np.isfinite(discount).all() or not discount.between(0, 1).all()
                or not np.isfinite(lead).all() or (lead < 0).any()):
            raise ValueError('Discount must map to [0,1]; lead_time must be finite and nonnegative.')
        keep = discount <= cfg.max_discount
        if cfg.max_lead is not None:
            if not np.isfinite(cfg.max_lead) or cfg.max_lead < 0:
                raise ValueError('max_lead must be finite and nonnegative.')
            keep &= lead <= cfg.max_lead
        audit['excluded_treatment_bounds'] = int((~keep).sum())
        d = d.loc[keep].copy()
        discount, lead = discount[keep], lead[keep]
        dl = [f'>{100*a:g}-{100*b:g}%' for a, b in zip(edges[:-1], edges[1:])]
        ll = ['0d', '>0-3d', '>3-7d', '>7-14d', '>14-30d', '>30d']
        db = pd.cut(discount, edges, labels=dl, include_lowest=True).astype(object)
        db.loc[discount == 0] = '0%'
        lb = pd.cut(lead, [-np.inf, 0, 3, 7, 14, 30, np.inf], labels=ll).astype(object)
        d['_action'] = lb.astype(str) + '|' + db.astype(str)
        all_actions = [a + '|' + b for a in ll for b in ['0%'] + dl]
        audit['discount_fraction_summary'] = {k: float(v) for k, v in discount.describe().items() if np.isfinite(v)}
    else:
        if d[cfg.treatment_col].isna().any():
            raise ValueError('Missing treatment labels.')
        d['_action'] = d[cfg.treatment_col]
        all_actions = list(cfg.categories) if cfg.categories is not None else sorted(pd.unique(d['_action']), key=str)
    if not 0 < cfg.sample_group_fraction <= 1:
        raise ValueError('sample_group_fraction must lie in (0,1].')
    if cfg.sample_group_fraction < 1:
        users = d[cfg.group_col].drop_duplicates().sample(frac=cfg.sample_group_fraction, random_state=cfg.random_state)
        d = d[d[cfg.group_col].isin(users)].copy()
    # Keep missing X values as an explicit reportable category. Encoders fit later.
    for col in cfg.x_cols:
        if not pd.api.types.is_numeric_dtype(d[col]):
            d[col] = d[col].astype('string').fillna('__missing__').astype(object)
    audit['eligible_rows'] = len(d)
    if len(d) < 30 or d[cfg.group_col].nunique() < max(6, cfg.cv + 2):
        raise ValueError('Insufficient rows or independent groups.')
    return d.reset_index(drop=True), all_actions, audit


def _encoder(df, cols):
    if not cols:
        return None
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    parts = []
    if numeric:
        parts.append(('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric))
    if categorical:
        parts.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__missing__', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.float32)),
        ]), categorical))
    return ColumnTransformer(parts, sparse_threshold=1.0)


def _clean_features(df, cols):
    d = df[cols].copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(d[col]):
            d[col] = d[col].astype('string').fillna('__missing__').astype(object)
        else:
            d[col] = pd.to_numeric(d[col]).replace([np.inf, -np.inf], np.nan)
    return d


@dataclass
class FittedPipeline:
    cfg: PipelineConfig
    estimator: CustomCausalDML
    x_encoder: object
    w_encoder: object

    def encode(self, df):
        x = self.x_encoder.transform(_clean_features(df, self.cfg.x_cols))
        w = None if self.w_encoder is None else self.w_encoder.transform(_clean_features(df, self.cfg.w_cols))
        return x.astype(np.float32), None if w is None else w.astype(np.float32)

    def effect(self, df, *, T0=None, T1=None):
        x, _ = self.encode(df)
        return self.estimator.effect(x, T0=T0, T1=T1)


def _fit(train, cfg, categories, final):
    xe, we = _encoder(train, cfg.x_cols), _encoder(train, cfg.w_cols)
    x = xe.fit_transform(_clean_features(train, cfg.x_cols)).astype(np.float32)
    w = None if we is None else we.fit_transform(_clean_features(train, cfg.w_cols)).astype(np.float32)
    defaults = dict(n_estimators=120, learning_rate=.05, num_leaves=15,
                    min_child_samples=100, max_bin=63, reg_lambda=2,
                    subsample=.8, subsample_freq=1, colsample_bytree=.8,
                    n_jobs=cfg.n_jobs, verbosity=-1, random_state=cfg.random_state)
    my = LGBMRegressor(**(defaults | cfg.nuisance_y_params))
    cls = LGBMClassifier if cfg.discrete_treatment else LGBMRegressor
    mt = cls(**(defaults | cfg.nuisance_t_params))
    model = CustomCausalDML(discrete_treatment=cfg.discrete_treatment, categories=categories,
                            model_y=my, model_t=mt,
                            cv=cfg.cv, final_config=final, n_jobs=cfg.n_jobs,
                            random_state=cfg.random_state)
    model.fit(train['_outcome'].to_numpy(), train['_action'].to_numpy(),
              X=x, W=w, groups=train[cfg.group_col].to_numpy())
    return FittedPipeline(cfg, model, xe, we)


def _diagnostic(bundle, df):
    x, w = bundle.encode(df)
    return bundle.estimator.residual_diagnostics(df['_outcome'].to_numpy(), df['_action'].to_numpy(), X=x, W=w)


def run_pipeline(df, cfg=None, output_dir='outputs'):
    """60/20/20 group split, validation selection, optional refit, one test report.

    Outcomes must already be mature and correctly attributed. For temporal
    deployment, replace the random group split with a purged temporal design.
    This function never fabricates input data or timestamps.
    """
    cfg = cfg or PipelineConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.final_candidates:
        raise ValueError('At least one final candidate is required.')
    d, all_actions, audit = prepare_data(df, cfg)
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=cfg.random_state)
    rest_i, test_i = next(first.split(d, groups=d[cfg.group_col]))
    rest = d.iloc[rest_i]
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=cfg.random_state+1)
    train_i, valid_i = next(second.split(rest, groups=rest[cfg.group_col]))
    train, valid, test = rest.iloc[train_i].copy(), rest.iloc[valid_i].copy(), d.iloc[test_i].copy()
    frames = {'train': train, 'validation': valid, 'test': test}
    group_sets = [set(f[cfg.group_col]) for f in frames.values()]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    if cfg.discrete_treatment:
        counts = train['_action'].value_counts()
        allowed = [a for a in all_actions if counts.get(a, 0) >= cfg.min_action_rows]
        if cfg.reference not in allowed:
            raise ValueError(f'Reference {cfg.reference!r} lacks train support. Inspect treatment units/bins.')
        categories = [cfg.reference] + [a for a in allowed if a != cfg.reference]
        if len(categories) < 2:
            raise ValueError('Fewer than two supported training actions.')
        preflight = pd.DataFrame({'action': all_actions})
        for name, f in frames.items():
            preflight[name+'_rows'] = preflight['action'].map(f['_action'].value_counts()).fillna(0).astype(int)
            frames[name] = f[f['_action'].isin(categories)].copy()
        preflight['included'] = preflight['action'].isin(categories)
        preflight.to_csv(out/'preflight_actions.csv', index=False)
    else:
        categories = None
    train, valid, test = frames.values()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('A split is empty after training-action support filtering.')
    audit['splits'] = {name: {'rows': len(f), 'groups': int(f[cfg.group_col].nunique())} for name, f in frames.items()}
    audit['note'] = 'Random group holdout. Review outcome maturity, post-treatment features, and temporal leakage separately.'
    scores, best, best_loss, selected = [], None, np.inf, None
    for i, final in enumerate(cfg.final_candidates):
        candidate = _fit(train, cfg, categories, final)
        diag = _diagnostic(candidate, valid)
        scores.append({'candidate': i, 'final_config': asdict(final), **diag})
        if diag['cate_residual_loss'] < best_loss:
            best, best_loss, selected = candidate, diag['cate_residual_loss'], final
    write_json(out/'validation_selection.json', scores)
    fit_data = pd.concat([train, valid]) if cfg.refit_train_validation else train
    if cfg.refit_train_validation:
        best = _fit(fit_data, cfg, categories, selected)
    report = {'test': _diagnostic(best, test), 'selected_final_config': asdict(selected),
              'identification': best.estimator.identification_, 'fold_audit': best.estimator.fold_audit_,
              'training_loss_history': best.estimator.loss_history_, 'data_audit': audit,
              'categories': categories, 'outcome': 'log1p' if cfg.log1p_outcome else 'original units'}
    write_json(out/'diagnostics.json', report)
    write_json(out/'config.json', asdict(cfg))
    joblib.dump(best, out/'fitted_pipeline.joblib', compress=3)
    if cfg.discrete_treatment:
        _reports(best, fit_data, test, categories, out)
    return best, report


def _reports(bundle, support_data, population, categories, out):
    """Action contrasts, cohort support, and scenarios on held-out X population."""
    cfg = bundle.cfg
    x, _ = bundle.encode(population)
    keys = cfg.x_cols
    support = support_data.groupby(keys+['_action'], observed=True, dropna=False).size().rename('fit_action_rows').reset_index()
    tables, scenario_rows = [], []
    for action in categories:
        effects = bundle.estimator.effect(x, T0=categories[0], T1=action)
        rows = population[keys].copy()
        rows['effect'] = effects
        tab = rows.groupby(keys, observed=True, dropna=False)['effect'].agg(['mean', 'size']).reset_index()
        tab = tab.rename(columns={'mean': 'mean_effect', 'size': 'evaluation_rows'})
        tab['_action'] = action
        tab = tab.merge(support, on=keys+['_action'], how='left')
        ref = support[support['_action'] == categories[0]].drop(columns='_action').rename(columns={'fit_action_rows': 'fit_reference_rows'})
        tab = tab.merge(ref, on=keys, how='left')
        tab[['fit_action_rows', 'fit_reference_rows']] = tab[['fit_action_rows', 'fit_reference_rows']].fillna(0).astype(int)
        tab['supported'] = ((tab['fit_action_rows'] >= cfg.min_cohort_action_rows)
                            & (tab['fit_reference_rows'] >= cfg.min_cohort_action_rows))
        tab['weak_global_identification'] = bundle.estimator.identification_['weak']
        if cfg.log1p_outcome:
            tab['shifted_geometric_pct'] = 100*np.expm1(tab['mean_effect'])
        tables.append(tab)
        for scenario in cfg.scenarios:
            mask = np.ones(len(population), dtype=bool)
            for col, values in scenario.get('filters', {}).items():
                if col not in population:
                    raise ValueError(f'Unknown scenario column: {col}')
                if values:
                    mask &= population[col].isin(values).to_numpy()
            val = float(effects[mask].mean()) if mask.any() else None
            scenario_rows.append(dict(name=scenario['name'], action=action, reference=categories[0],
                                      evaluation_rows=int(mask.sum()), mean_effect=val,
                                      shifted_geometric_pct=100*np.expm1(val) if val is not None and cfg.log1p_outcome else None))
    cohort = pd.concat(tables, ignore_index=True)
    cohort.to_csv(out/'cohort_effects.csv', index=False)
    pd.DataFrame(scenario_rows).to_csv(out/'scenario_effects.csv', index=False)
    support.to_csv(out/'cohort_action_support.csv', index=False)
    # Descriptive model contrasts only: not a doubly robust policy-value estimator.
    good = cohort[cohort['supported'] & ~cohort['weak_global_identification']]
    opportunities = good.sort_values('mean_effect', ascending=False).groupby(keys, observed=True, dropna=False).head(1)
    opportunities = opportunities[opportunities['mean_effect'] > 0]
    opportunities.to_csv(out/'exploratory_opportunities.csv', index=False)
    _plot(cohort, keys, out, cfg.log1p_outcome)
    if cfg.treatment_col is None:
        _mitigation(bundle, x, categories, out)


def _plot(cohort, keys, out, log_outcome):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tab = cohort.copy()
    tab['cohort'] = tab[keys].astype(str).agg(' | '.join, axis=1)
    selected = tab.groupby('cohort')['evaluation_rows'].max().nlargest(20).index
    value = 'shifted_geometric_pct' if log_outcome else 'mean_effect'
    tab.loc[~tab['supported'], value] = np.nan
    matrix = tab[tab['cohort'].isin(selected)].pivot(index='cohort', columns='_action', values=value)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, .7*len(matrix.columns)), max(3, .4*len(matrix))))
    limit = np.nanmax(np.abs(matrix.to_numpy())) if np.isfinite(matrix.to_numpy()).any() else 1.
    im = ax.imshow(matrix, cmap='RdBu', vmin=-max(limit, 1e-8), vmax=max(limit, 1e-8), aspect='auto')
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=75, ha='right')
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title('Action vs reference: model contrasts; blank = insufficient support')
    fig.colorbar(im, ax=ax, label='Shifted geometric outcome change (%)' if log_outcome else 'Outcome effect')
    fig.tight_layout()
    fig.savefig(out/'cohort_heatmap.png', dpi=140)
    plt.close(fig)


def _mitigation(bundle, x, categories, out):
    cfg = bundle.cfg
    rows = []
    if '0d|0%' not in categories:
        write_json(out/'mitigation_unavailable.json', {'reason': 'Zero-lead zero-discount reference is not supported.'})
        return
    for action in categories:
        lead, discount = action.split('|')
        no_discount = lead+'|0%'
        if no_discount not in categories:
            continue
        loss = bundle.estimator.ate(x, T0='0d|0%', T1=no_discount)
        recover = bundle.estimator.ate(x, T0=no_discount, T1=action)
        # For log1p Y this compares geometric means of Y+1; arithmetic revenue is not identified.
        penalty = -np.expm1(loss) if cfg.log1p_outcome else -loss
        gain = np.exp(loss)*np.expm1(recover) if cfg.log1p_outcome else recover
        rows.append(dict(action=action, same_lead_reference=no_discount,
                         lead_effect=loss, discount_effect_at_fixed_lead=recover,
                         recovery_ratio=gain/penalty if penalty > 1e-8 else None,
                         note='Exploratory; consult cohort support. Ratio undefined without a lead penalty.'))
    pd.DataFrame(rows).to_csv(out/'lead_discount_mitigation.csv', index=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Fit on real CSV data using editable PipelineConfig defaults.')
    p.add_argument('csv')
    p.add_argument('--output', default='outputs')
    args = p.parse_args()
    run_pipeline(pd.read_csv(args.csv), output_dir=args.output)
