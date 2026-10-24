from dataclasses import replace
import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from causal_dml import LightweightDRLearner, FinalStageConfig
from recommendation_pipeline import RecommendationConfig, run_recommendations, validate_splits


FAST = FinalStageConfig(n_estimators=50, num_leaves=7, max_depth=3,
    min_child_samples=30, reg_lambda=.5, extra_trees=False)


def data(n=3000, seed=2026):
    rng = np.random.default_rng(seed)
    x, w = rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    t = rng.binomial(1, .4, n)
    tau = 1+.7*x[:, 0]
    y = w[:, 0]+t*tau+rng.normal(0, .2, n)
    return y, t, x, w, tau


def test_DR_known_randomization_heterogeneity_and_serialization(tmp_path):
    y, t, x, w, tau = data()
    m = LightweightDRLearner(categories=[0, 1], known_propensities=[.6, .4],
        model_y=LinearRegression(), final_config=FAST).fit(y, t, X=x, W=w)
    assert m.const_marginal_effect(x).shape == (len(y), 1)
    assert np.mean(abs(m.effect(x)-tau)) < .3
    assert_allclose(m.effect(x, T0=1, T1=0), -m.effect(x))
    assert m.model_t_ is None
    joblib.dump(m, tmp_path/'dr.joblib')
    assert_allclose(joblib.load(tmp_path/'dr.joblib').effect(x), m.effect(x))


def test_DR_propensity_correct_outcome_wrong():
    y, t, x, w, _ = data(14000)
    m = LightweightDRLearner(categories=[0, 1], known_propensities=[.6, .4],
        model_y=DummyRegressor(), final_config=FAST).fit(y, t, W=w)
    assert abs(m.ate()-1) < .1
    assert len(m.recommend(None, W=w)['action']) == len(y)


def test_DR_outcome_correct_propensity_wrong():
    rng = np.random.default_rng(1)
    n = 5000
    w = rng.normal(size=(n, 1))
    t = rng.binomial(1, 1/(1+np.exp(-w[:, 0])))
    y = 3*w[:, 0]+1.75*t
    m = LightweightDRLearner(categories=[0, 1], model_y=LinearRegression(),
        model_t=DummyClassifier(strategy='uniform')).fit(y, t, W=w)
    assert_allclose(m.ate(), 1.75, atol=1e-4)


def test_DR_multiarm_W_modifiers_are_averaged_in_final_X():
    rng = np.random.default_rng(3)
    n = 7000
    x = rng.normal(size=(n, 1))
    w = rng.normal(size=(n, 1))
    t = rng.choice(['r', 'a', 'b'], n, p=[.2, .5, .3])
    y = w[:, 0]+(t == 'a')*(2+.3*w[:, 0])-(t == 'b')
    m = LightweightDRLearner(categories=['r', 'a', 'b'], known_propensities=[.2,.5,.3],
        model_y=DummyRegressor(), final_config=FAST).fit(y, t, W=w)
    assert_allclose(m.const_marginal_effect(), [[2, -1]], atol=.1)
    assert_allclose(m.effect(T0='a', T1='b'), [-3], atol=.15)


def test_DR_fixed_policy_evaluation_cluster_interval_and_cost():
    y, t, x, w, _ = data(2400)
    groups = np.repeat(np.arange(800), 3)
    m = LightweightDRLearner(categories=[0, 1], known_propensities=[.6,.4],
        model_y=LinearRegression(), final_config=FAST).fit(y, t, X=x, W=w, groups=groups)
    ey, et, ex, ew, _ = data(3000, 8)
    g = np.repeat(np.arange(1000)+10000, 3)
    report = m.evaluate_policy(ey, et, X=ex, W=ew, policy=1, baseline=0, groups=g, costs=[0,.2])
    assert abs(report['mean_gain']-.8) < .15
    assert report['ci_low'] <= report['mean_gain'] <= report['ci_high']
    assert report['independent_units'] == 1000 and report['overlap_pass']
    same = m.evaluate_policy(ey, et, X=ex, W=ew, policy=0, baseline=0, groups=g)
    assert same['mean_gain'] == same['standard_error'] == 0
    with pytest.raises(ValueError, match='overlap'):
        m.evaluate_policy(y, t, X=x, W=w, policy=1, groups=groups)


def test_DR_recommendation_eligibility_and_costs():
    y, t, x, w, _ = data(1200)
    m = LightweightDRLearner(categories=[0,1], known_propensities=[.6,.4],
        final_config=FAST).fit(y,t,X=x,W=w)
    no = m.recommend(x, W=w, costs=[0,100])
    assert (no['action'] == 0).all()
    eligible = np.ones((len(y),2),dtype=bool)
    eligible[:,1] = False
    assert (m.recommend(x,W=w,eligible=eligible)['action'] == 0).all()
    eligible[:,0] = False
    assert not m.recommend(x,W=w,eligible=eligible)['supported'].any()


def test_DR_independent_calibration_and_heldout_score():
    y,t,x,w,_ = data(1500)
    m = LightweightDRLearner(categories=[0,1], final_config=FAST,
        known_propensities=[.6,.4]).fit(y,t,X=x,W=w,groups=np.arange(len(y)))
    cy,ct,cx,cw,_ = data(800,11)
    cg = np.arange(800)+10000
    m.fit_calibration(cy,ct,X=cx,W=cw,groups=cg)
    with pytest.raises(ValueError,match='overlap'):
        m.dr_diagnostics(cy,ct,X=cx,W=cw,groups=cg)
    ty,tt,tx,tw,_ = data(700,12)
    diag = m.dr_diagnostics(ty,tt,X=tx,W=tw,groups=np.arange(700)+20000)
    assert np.isfinite(diag['dr_pseudo_mse'])
    with pytest.raises(ValueError,match='already'):
        m.fit_calibration(cy,ct,X=cx,W=cw,groups=cg)


def test_DR_failed_refit_and_malformed_shapes():
    y,t,x,w,_ = data(500)
    m = LightweightDRLearner(categories=[0,1], final_config=FAST).fit(y,t,X=x,W=w)
    with pytest.raises(ValueError):
        m.fit(y,np.column_stack([t,t]),X=x,W=w)
    with pytest.raises(RuntimeError):
        m.effect(x)


def test_complete_recommendation_pipeline_with_product_attributes(tmp_path):
    y,t,x,w,_ = data(1800)
    df = pd.DataFrame(dict(revenue=np.maximum(y+5,0),user_id=np.repeat(np.arange(600),3),
        product_id=np.arange(len(y))%1000, industry=np.where(x[:,0]>0,'i','j'),
        data_definition='lab_water', day=w[:,0], carousel=t, emb0=x[:,0]))
    cfg = RecommendationConfig(treatment_col='carousel', reference=0, categories=[0,1],
        w_cols=['day'], embedding_cols=['emb0'], product_col='product_id',
        known_propensities=[.6,.4], min_action_rows=5, min_cohort_action_rows=5,
        cv=2, calibrate=True, final_candidates=[FAST], scenarios=[dict(name='all',filters={})])
    model, report = run_recommendations(df,cfg,tmp_path)
    assert (tmp_path/'recommendations_test_population.csv').is_file()
    assert report['outcome_scale'] == 'original'
    assert report['policy_evaluation']['independent_units'] > 0
    assert 'product_id' not in model.cfg.x_cols
    assert model.estimator.calibrators_[0] is not None
    rec = pd.read_csv(tmp_path/'recommendations_test_population.csv')
    assert 'product_id' in rec


def test_custom_split_groups_leakage():
    split = dict(train=np.arange(3),selection=np.arange(3,6),calibration=np.arange(6,9),test=np.arange(9,12))
    validate_splits(split,12,np.arange(12))
    with pytest.raises(ValueError,match='leakage'):
        validate_splits(split,12,np.zeros(12))
