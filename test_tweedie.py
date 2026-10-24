"""Controlled implementation checks; notebooks use the user's actual data."""
import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import clone

from causal_dml import TweedieMeanRegressor, LightweightDRLearner, FinalStageConfig
from recommendation_pipeline import RecommendationConfig, run_recommendations, validate_revenue_config


PARAMS = dict(n_estimators=40, num_leaves=7, min_child_samples=20,
              learning_rate=.08, n_jobs=2, verbosity=-1)


def revenue_data(n=2400):
    rng = np.random.default_rng(51)
    x = rng.normal(size=(n,2)).astype('float32')
    t = rng.binomial(1,.5,n)
    # Zero-inflated continuous positive revenue; not application/demo data.
    y = rng.binomial(1,.3,n)*rng.gamma(2,5*np.exp(.2*x[:,0]+.2*t))
    return x,t,y


def test_tweedie_log_link_original_unit_predictions_and_clone(tmp_path):
    x,_,y = revenue_data()
    m = clone(TweedieMeanRegressor(variance_power=1.5, model_params=PARAMS)).fit(x,y)
    p = m.predict(x)
    assert np.isfinite(p).all() and (p > 0).all()
    assert m.model_.get_params()['objective'] == 'tweedie'
    assert_allclose(np.exp(m.model_.predict(x,raw_score=True)),p)
    assert_allclose(m.predict_log_mean(x),m.model_.predict(x,raw_score=True))
    assert abs(p.mean()-y.mean())/y.mean() < .2
    joblib.dump(m,tmp_path/'tweedie.joblib')
    assert_allclose(joblib.load(tmp_path/'tweedie.joblib').predict(x),p)


@pytest.mark.parametrize('power',[1.,2.,float('nan')])
def test_tweedie_invalid_power(power):
    with pytest.raises(ValueError,match='between'):
        TweedieMeanRegressor(variance_power=power).fit(np.ones((10,1)),np.arange(10))


def test_tweedie_zero_fit_sparse_and_failed_refit():
    x = sparse.csr_matrix(np.ones((30,2)))
    m = TweedieMeanRegressor().fit(x,np.zeros(30))
    assert_allclose(m.predict(x),0)
    assert m.model_ is None
    with pytest.raises(ValueError,match='nonnegative'):
        m.fit(x,-np.ones(30))
    with pytest.raises(RuntimeError):
        m.predict(x)


def test_tweedie_pipeline_rejects_invalid_scale_and_all_zero():
    df = pd.DataFrame({'revenue':[0.,1.,2.]})
    with pytest.raises(ValueError,match='original-unit'):
        validate_revenue_config(df,RecommendationConfig(log1p_outcome=True))
    with pytest.raises(ValueError,match='nonnegative'):
        validate_revenue_config(df.assign(revenue=[-1,0,1]),RecommendationConfig())
    with pytest.raises(ValueError,match='all zero'):
        validate_revenue_config(df.assign(revenue=0),RecommendationConfig())
    validate_revenue_config(df,RecommendationConfig(outcome_nuisance='regression',log1p_outcome=True))


def test_tweedie_full_DR_pipeline_reports_currency_and_signed_effects(tmp_path):
    x,t,y = revenue_data(2400)
    df = pd.DataFrame(dict(revenue=y,user_id=np.repeat(np.arange(800),3),
        data_definition=np.where(x[:,0]>0,'water','salt'),industry='lab',
        day=x[:,1],carousel=t))
    cfg = RecommendationConfig(treatment_col='carousel',categories=[0,1],reference=0,
        w_cols=['day'],known_propensities=[.5,.5],min_action_rows=5,
        min_cohort_action_rows=5,cv=2,nuisance_y_params=PARAMS,
        final_candidates=[FinalStageConfig(n_estimators=30,num_leaves=7,max_depth=3,min_child_samples=30)],
        scenarios=[dict(name='all',filters={})])
    fitted, report = run_recommendations(df,cfg,tmp_path)
    assert isinstance(fitted.estimator.model_y_,TweedieMeanRegressor)
    assert report['outcome_link'] == 'log' and report['currency'] == 'EUR'
    assert (tmp_path/'policy_revenue_summary.csv').is_file()
    xe,_ = fitted.encode(df)
    forward = fitted.estimator.effect(xe,T0=0,T1=1)
    assert_allclose(fitted.estimator.effect(xe,T0=1,T1=0),-forward)
    for m in fitted.estimator.models_:
        if m is not None:
            assert m.params['objective'] == 'regression'  # signed pseudo-targets, not Tweedie
    summary=pd.read_csv(tmp_path/'policy_revenue_summary.csv').iloc[0]
    assert_allclose(summary.policy_mean_value-summary.baseline_mean_value,summary.incremental_value_per_opportunity)


def test_policy_percentage_is_ratio_of_mean_values():
    x,t,y = revenue_data(1600)
    # Add a positive reference mean so ratio interpretation is well-conditioned.
    y = 10+2*t
    m = LightweightDRLearner(categories=[0,1],known_propensities=[.5,.5],
        model_y=TweedieMeanRegressor(model_params=PARAMS)).fit(y,t)
    heldout_t = np.random.default_rng(52).binomial(1,.5,2000)
    heldout_y = 10+2*heldout_t
    result=m.evaluate_policy(heldout_y,heldout_t,policy=1)
    assert abs(result['baseline_mean_value']-10) < .1
    assert abs(result['policy_mean_value']-12) < .1
    assert abs(result['relative_gain_pct']-20) < 1
    assert_allclose(result['relative_gain_pct'],100*result['mean_gain']/result['baseline_mean_value'])
