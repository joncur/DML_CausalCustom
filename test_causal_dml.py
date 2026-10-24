"""Controlled fixtures test implementation; the application uses real data only."""
from dataclasses import replace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold

from causal_dml import CustomCausalDML, FinalStageConfig, orthogonal_derivatives
from pipeline import PipelineConfig, prepare_data, run_pipeline


FAST = FinalStageConfig(n_estimators=65, num_leaves=7, max_depth=3,
                        min_child_samples=35, min_child_weight=.05,
                        reg_lambda=.2, extra_trees=False, backfit_rounds=3)


def fixture(n=3000, seed=17):
    rng = np.random.default_rng(seed)
    x, w = rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    t = rng.binomial(1, .5, n)
    tau = 1.5 + .7*x[:, 0]
    y = .8*w[:, 0] + .3*x[:, 1] + t*tau + rng.normal(0, .12, n)
    return y, t, x, w, tau


def fit_binary(n=3000):
    y, t, x, w, tau = fixture(n)
    m = CustomCausalDML(categories=[0, 1], model_y=LinearRegression(),
                        model_t=DummyClassifier(strategy='prior'), final_config=FAST)
    m.fit(y, t, X=x, W=w)
    return m, (y, t, x, w, tau)


def test_custom_derivatives_finite_differences_and_zero_residual():
    rng = np.random.default_rng(10)
    r, partial, f = rng.normal(size=(3, 10))
    r[0] = 0
    g, h = orthogonal_derivatives(f, partial, r)
    eps = 1e-4
    loss = lambda z: .5*(partial-r*z)**2
    assert_allclose(g, (loss(f+eps)-loss(f-eps))/(2*eps), atol=1e-9)
    assert_allclose(h, (loss(f+eps)-2*loss(f)+loss(f-eps))/eps**2, atol=2e-7)
    assert g[0] == h[0] == 0


def test_coordinate_objective_is_exact_joint_loss():
    rng = np.random.default_rng(22)
    r = rng.normal(size=(70, 3))
    theta, new = rng.normal(size=(2, 70, 3))
    y = rng.normal(size=70)
    j = 1
    partial = y-(r*theta).sum(axis=1)+r[:, j]*theta[:, j]
    updated = theta.copy()
    updated[:, j] = new[:, j]
    assert_allclose((y-(r*updated).sum(axis=1))**2, (partial-r[:, j]*new[:, j])**2)


def test_binary_heterogeneity_and_axes():
    m, (_, _, x, _, tau) = fit_binary()
    pred = m.const_marginal_effect(x)
    assert pred.shape == (len(x), 1)
    assert np.mean(np.abs(pred[:, 0]-tau)) < .25
    assert_allclose(m.effect(x, T0=0, T1=1), pred[:, 0])
    assert_allclose(m.effect(x, T0=1, T1=0), -pred[:, 0])
    assert_allclose(m.effect(x, T0=1, T1=1), 0)
    assert_allclose(m.ate(x), pred.mean())
    assert np.all(np.diff(m.loss_history_) <= 1e-12)


def test_multiarm_joint_constant_recovers_signs_and_reference():
    rng = np.random.default_rng(5)
    n = 9000
    w = rng.normal(size=(n, 1))
    t = rng.choice(['base', 'a', 'b'], n, p=[.2, .5, .3])
    y = 2*w[:, 0] + (t == 'a')*1.2 - (t == 'b')*.8 + rng.normal(0, .15, n)
    m = CustomCausalDML(categories=['base', 'a', 'b'], model_y=LinearRegression(),
                        model_t=DummyClassifier(strategy='prior')).fit(y, t, W=w)
    assert_allclose(m.const_marginal_effect(), [[1.2, -.8]], atol=.03)
    assert_allclose(m.effect(T0='a', T1='b'), [-2.], atol=.04)
    assert m.categories_[0] == 'base'
    other = CustomCausalDML(categories=['b', 'base', 'a'], model_y=LinearRegression(),
                            model_t=DummyClassifier(strategy='prior')).fit(y, t, W=w)
    assert_allclose(other.effect(T0='a', T1='b'), m.effect(T0='a', T1='b'), atol=1e-6)


def test_multiarm_heterogeneous_coupled_stage():
    rng = np.random.default_rng(18)
    n = 5000
    x = rng.uniform(-1, 1, (n, 2))
    t = rng.choice(['r', 'a', 'b'], n, p=[.2, .5, .3])
    a, b = 1+.7*x[:, 0], -.8+.4*x[:, 1]
    y = .3*x[:, 0] + (t == 'a')*a + (t == 'b')*b + rng.normal(0, .1, n)
    m = CustomCausalDML(categories=['r', 'a', 'b'], model_y=LinearRegression(),
                        model_t=DummyClassifier(strategy='prior'), final_config=FAST).fit(y, t, X=x)
    p = m.const_marginal_effect(x)
    assert p.shape == (n, 2)
    assert np.mean(abs(p[:, 0]-a)) < .22
    assert np.mean(abs(p[:, 1]-b)) < .22
    assert_allclose(m.effect(x, T0='a', T1='b'), p[:, 1]-p[:, 0])


def test_continuous_slope_scaling():
    rng = np.random.default_rng(99)
    n = 3000
    x, w = rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    t = .6*w[:, 0]+rng.normal(size=n)
    tau = -.7+.4*x[:, 0]
    y = w[:, 0]+t*tau+rng.normal(0, .1, n)
    m = CustomCausalDML(discrete_treatment=False, model_y=lgb.LGBMRegressor(
        n_estimators=80, num_leaves=15, n_jobs=2, verbosity=-1), model_t=LinearRegression(), final_config=FAST)
    m.fit(y, t, X=x, W=w)
    p = m.const_marginal_effect(x)
    assert np.mean(abs(p[:, 0]-tau)) < .25
    assert_allclose(m.effect(x, T0=2, T1=4), p[:, 0]*2)
    with pytest.raises(ValueError):
        m.effect(x, T1=np.inf)


def test_group_splitter_and_disk_cleanup(tmp_path):
    y, t, x, w, _ = fixture(600)
    g = np.repeat(np.arange(100), 6)
    m = CustomCausalDML(categories=[0, 1], cv=GroupKFold(3), work_dir=tmp_path,
                        model_y=LinearRegression(), model_t=DummyClassifier(), final_config=FAST)
    m.fit(y, t, X=x, W=w, groups=g)
    assert all(a['shared_groups'] == 0 for a in m.fold_audit_)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('kind', ['groups', 'fractional', 'duplicate', 'overlap', 'incomplete'])
def test_bad_folds_rejected_before_fit(kind):
    y, t, x, w, _ = fixture(60)
    tr, va = np.arange(30), np.arange(30, 60)
    g = None
    if kind == 'groups':
        g = np.tile(np.arange(30), 2)
    if kind == 'fractional':
        tr = tr + .5
    if kind == 'duplicate':
        tr = np.r_[tr, 0]
    if kind == 'overlap':
        va = np.r_[va, 0]
    folds = [(tr, va)] if kind == 'incomplete' else [(tr, va), (va, tr)]
    with pytest.raises(ValueError):
        CustomCausalDML(categories=[0, 1], cv=folds).fit(y, t, X=x, W=w, groups=g)


@pytest.mark.parametrize('field', ['Y', 'T', 'W', 'groups'])
def test_malformed_inputs(field):
    y, t, x, w, _ = fixture(100)
    args = dict(Y=y, T=t, X=x, W=w, groups=np.arange(100))
    args[field] = np.ones((100, 2)) if field != 'W' else w[:2]
    with pytest.raises(ValueError):
        CustomCausalDML(categories=[0, 1]).fit(**args)


def test_failed_refit_disables_prediction_and_dataframe_schema():
    m, (y, t, x, w, _) = fit_binary(400)
    m.fit(y, t, X=pd.DataFrame(x, columns=['a', 'b']), W=w)
    with pytest.raises(ValueError):
        m.effect(pd.DataFrame(x, columns=['b', 'a']))
    with pytest.raises(ValueError):
        m.fit(y, np.column_stack([t, t]), X=x)
    with pytest.raises(RuntimeError):
        m.effect(x)


def test_batch_rowwise_contrasts_serialization_and_diagnostics(tmp_path):
    m, (y, t, x, w, _) = fit_binary(600)
    hi = np.arange(len(x)) % 2
    expected = m.effect(x, T0=0, T1=hi)
    assert_allclose(np.concatenate(list(m.iter_effect(x, T0=0, T1=hi, batch_size=73))), expected)
    joblib.dump(m, tmp_path/'model.joblib')
    restored = joblib.load(tmp_path/'model.joblib')
    assert_allclose(restored.effect(x), m.effect(x))
    diag = m.residual_diagnostics(y, t, X=x, W=w)
    assert m.score(y, t, X=x, W=w) == -diag['cate_residual_loss']
    with pytest.raises(ValueError):
        m.residual_diagnostics([1], t, X=x, W=w)
    with pytest.raises(ValueError):
        m.effect(x, T1=4)


def test_default_lightgbm_nuisances_and_weak_identification():
    y, t, x, w, _ = fixture(700)
    m = CustomCausalDML(categories=[0, 1], final_config=FAST).fit(y, t, X=x, W=w)
    assert np.isfinite(m.effect(x)).all()
    assert m.identification_['min_eigenvalue'] > 0


def test_explicit_constant_X_uses_joint_constant_final():
    y, t, x, w, _ = fixture(500)
    constant_x = np.ones((len(y), 2), dtype='float32')
    m = CustomCausalDML(categories=[0, 1], final_config=FAST).fit(y, t, X=constant_x, W=w)
    assert m.final_stage_constant_
    assert_allclose(m.effect(constant_x), m.constant_effect_[0])


def test_confounding_without_effect_is_removed():
    rng = np.random.default_rng(28)
    n = 5000
    w = rng.normal(size=(n, 1))
    x = rng.normal(size=(n, 1))
    t = rng.binomial(1, 1/(1+np.exp(-w[:, 0])))
    y = 3*w[:, 0]+rng.normal(0, .2, n)
    naive = y[t == 1].mean()-y[t == 0].mean()
    assert naive > 1
    m = CustomCausalDML(categories=[0, 1], model_y=LinearRegression(),
                        model_t=LogisticRegression(), final_config=FAST).fit(y, t, X=x, W=w)
    assert abs(m.ate(x)) < .08


def test_bin_units_and_bounds():
    n = 60
    df = pd.DataFrame(dict(revenue=np.arange(n), user_id=np.arange(n), industry='i',
        data_definition='lab_water', region='r', discount_percentage=np.tile([0, 1, 3, 6, 45, 60], 10), lead_time=0))
    cfg = PipelineConfig(w_cols=[])
    percent, _, audit = prepare_data(df, cfg)
    assert audit['excluded_treatment_bounds'] == 10
    fraction, _, _ = prepare_data(df.assign(discount_percentage=df.discount_percentage/100), replace(cfg, discount_input_is_percent=False))
    assert percent['_action'].tolist() == fraction['_action'].tolist()
    assert percent['_action'].iloc[:4].tolist() == ['0d|0%', '0d|>0-1%', '0d|>1-3%', '0d|>3-6%']


def test_complete_real_data_interface_and_training_only_preprocessing(tmp_path):
    # Deliberately controlled input for an integration test, never application demo data.
    y, t, x, w, _ = fixture(1800)
    df = pd.DataFrame(dict(revenue=np.maximum(0, y+4), user_id=np.repeat(np.arange(600), 3),
        industry=np.where(x[:, 0] > 0, 'a', 'b'), data_definition='lab_water', region='r',
        day=w[:, 0], carousel=t))
    cfg = PipelineConfig(treatment_col='carousel', reference=0, categories=[0, 1],
                         w_cols=['day'], min_action_rows=10, min_cohort_action_rows=10,
                         final_candidates=[FAST], cv=2)
    model, report = run_pipeline(df, cfg, tmp_path)
    assert (tmp_path/'cohort_effects.csv').is_file()
    assert (tmp_path/'fitted_pipeline.joblib').is_file()
    assert np.isfinite(model.effect(df, T0=0, T1=1)).all()
    assert len(report['fold_audit']) == 2
    unseen = df.iloc[:4].assign(industry='unseen')
    assert np.isfinite(model.effect(unseen)).all()
    assert not (tmp_path/'lead_discount_mitigation.csv').exists()


def test_joint_pipeline_grid_mitigation_and_reference_support(tmp_path):
    y, t, x, w, _ = fixture(1500)
    rng = np.random.default_rng(8)
    arm = rng.integers(0, 3, len(y))
    df = pd.DataFrame(dict(revenue=np.maximum(y+5, 0), user_id=np.repeat(np.arange(500), 3),
        industry=np.where(x[:, 0] > 0, 'i', 'j'), data_definition='lab_water', region='r',
        discount_percentage=np.where(arm == 2, 5., 0.), lead_time=np.where(arm == 0, 0., 3.)))
    cfg = PipelineConfig(w_cols=[], min_action_rows=5, min_cohort_action_rows=5,
                         cv=2, final_candidates=[replace(FAST, n_estimators=15)])
    model, report = run_pipeline(df, cfg, tmp_path)
    grid = pd.read_csv(tmp_path/'preflight_actions.csv')
    assert len(grid) == 42 and len(grid[grid.train_rows == 0]) > 0
    mitigation = pd.read_csv(tmp_path/'lead_discount_mitigation.csv')
    assert '>0-3d|>3-6%' in mitigation.action.to_list()
    assert report['categories'][0] == '0d|0%'
