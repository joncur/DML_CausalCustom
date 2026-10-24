"""Self-contained cross-fitted R-learner with a coupled LightGBM final stage.

Single outcome; categorical or scalar continuous treatment. Numeric encoded
X/W only. No EconML dependency, forest inference, or project-specific imports.
"""
from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.utils.validation import check_is_fitted


def _vector(value, name):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1 or not len(a):
        raise ValueError(f'{name} must be a nonempty vector or one column.')
    return a


def _numeric(value, name):
    if sparse.issparse(value):
        a = value.tocsr().astype(np.float32, copy=False)
        ok = np.isfinite(a.data).all()
    else:
        a = np.asarray(value, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        ok = np.isfinite(a).all()
    if a.ndim != 2 or a.shape[1] < 1 or not ok:
        raise ValueError(f'{name} must be a finite, pre-encoded numeric matrix.')
    return a


def _combine(x, w):
    if w is None:
        return x
    if sparse.issparse(x) or sparse.issparse(w):
        return sparse.hstack([x, w], format='csr', dtype=np.float32)
    return np.concatenate([x, w], axis=1)


def _has_variation(x):
    if sparse.issparse(x):
        return bool(np.any(x.max(axis=0).toarray() != x.min(axis=0).toarray()))
    return bool(np.any(x.max(axis=0) != x.min(axis=0)))


def _pred_vector(model, x):
    p = np.asarray(model.predict(x), dtype=float)
    if p.shape == (x.shape[0], 1):
        p = p[:, 0]
    if p.shape != (x.shape[0],) or not np.isfinite(p).all():
        raise ValueError('Nuisance predict must return one finite value per row.')
    return p


def orthogonal_derivatives(pred, partial, residual):
    """Exact derivatives of 0.5*(partial - residual*pred)**2, no division."""
    return residual * (residual * pred - partial), residual ** 2


class TweedieMeanRegressor(BaseEstimator):
    """LightGBM Tweedie outcome nuisance with log link and original-unit means.

    Fits nonnegative Y directly, not log1p(Y). Power in (1,2) targets a
    compound-Poisson/Gamma mean-variance family with a mass at zero. This is
    mean regression, not a fitted full predictive distribution or an outlier-
    resistant estimator. Never use it on signed DR contrasts/residuals.
    """
    def __init__(self, *, variance_power=1.5, model_params=None):
        self.variance_power = variance_power
        self.model_params = model_params

    def fit(self, X, y):
        self.is_fitted_ = False
        if not np.isfinite(self.variance_power) or not 1 < self.variance_power < 2:
            raise ValueError('variance_power must be strictly between 1 and 2.')
        x = _numeric(X, 'X')
        y = _vector(y, 'y').astype(float)
        if len(y) != x.shape[0] or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Tweedie revenue labels must be finite, nonnegative, original-unit values.')
        params = dict(n_estimators=120, learning_rate=.05, num_leaves=15,
                      min_child_samples=100, max_bin=63, reg_lambda=2,
                      n_jobs=2, random_state=2026, verbosity=-1)
        params.update(self.model_params or {})
        for alias in ('objective', 'application', 'objective_type', 'app'):
            if alias in params and params.pop(alias) != 'tweedie':
                raise ValueError('TweedieMeanRegressor cannot override its Tweedie objective.')
        if 'tweedie_variance_power' in params and params.pop('tweedie_variance_power') != self.variance_power:
            raise ValueError('Specify variance_power once; conflicting Tweedie power settings.')
        self.n_features_in_ = x.shape[1]
        self.zero_only_ = bool(np.all(y == 0))
        self.model_ = None
        if not self.zero_only_:
            self.model_ = lgb.LGBMRegressor(objective='tweedie',
                tweedie_variance_power=self.variance_power, **params).fit(x, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError('Tweedie nuisance fit must complete successfully first.')
        x = _numeric(X, 'X')
        if x.shape[1] != self.n_features_in_:
            raise ValueError('Tweedie nuisance X width changed.')
        p = np.zeros(x.shape[0]) if self.zero_only_ else _pred_vector(self.model_, x)
        if (p < 0).any():
            raise ValueError('Tweedie mean prediction must be nonnegative.')
        return p

    def predict_log_mean(self, X):
        """Log link linear predictor; log(0)=-inf for the degenerate all-zero fit."""
        with np.errstate(divide='ignore'):
            return np.log(self.predict(X))


@dataclass(frozen=True)
class FinalStageConfig:
    """Bounded, stochastic boosting; not LightGBM's rf mode or an honest forest."""
    n_estimators: int = 160
    learning_rate: float = 0.05
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 200
    min_child_weight: float = 1.0
    subsample: float = 0.7
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    reg_alpha: float = 0.0
    extra_trees: bool = True
    max_bin: int = 63
    backfit_rounds: int = 3
    tolerance: float = 1e-5

    def validate(self):
        for name in ('n_estimators', 'num_leaves', 'max_depth',
                     'min_child_samples', 'max_bin', 'backfit_rounds'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.num_leaves < 2 or self.max_bin < 2:
            raise ValueError('num_leaves and max_bin must be at least 2.')
        for name in ('learning_rate', 'subsample', 'colsample_bytree'):
            v = getattr(self, name)
            if not np.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f'{name} must lie in (0,1].')
        for name in ('min_child_weight', 'reg_lambda', 'reg_alpha', 'tolerance'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')


class CustomCausalDML(BaseEstimator):
    """Reusable estimator with explicit treatment and output-axis conventions.

    Both nuisances see [X,W]; final effects see X only. With X=None the final
    stage is a constant joint residual regression. With groups, cross-fitting
    never puts a group on both sides. groups does not implement inference.

    const_marginal_effect: (n,q), including binary/continuous q=1.
    effect: (n,), always T1 minus T0. ate: empirical mean of requested effects.
    """
    def __init__(self, *, discrete_treatment=True, categories=None, cv=3,
                 model_y=None, model_t=None, final_config=None, ridge=1e-8,
                 identification_tol=1e-7, random_state=2026, n_jobs=2,
                 batch_size=20000, work_dir=None):
        self.discrete_treatment = discrete_treatment
        self.categories = categories
        self.cv = cv
        self.model_y = model_y
        self.model_t = model_t
        self.final_config = final_config
        self.ridge = ridge
        self.identification_tol = identification_tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.work_dir = work_dir

    def _nuisance(self, treatment=False):
        given = self.model_t if treatment else self.model_y
        if given is not None:
            return clone(given)
        cls = lgb.LGBMClassifier if treatment and self.discrete_treatment else lgb.LGBMRegressor
        return cls(n_estimators=120, learning_rate=.05, num_leaves=15,
                   min_child_samples=100, max_bin=63, reg_lambda=2,
                   subsample=.8, subsample_freq=1, colsample_bytree=.8,
                   n_jobs=self.n_jobs, verbosity=-1, random_state=self.random_state)

    def _design(self, t):
        if not self.discrete_treatment:
            t = np.asarray(t, dtype=float)
            if not np.isfinite(t).all():
                raise ValueError('Continuous T must be finite.')
            return t[:, None]
        try:
            ids = np.array([self._category_index_[v] for v in t])
        except (KeyError, TypeError) as e:
            raise ValueError('Unknown treatment label.') from e
        return (ids[:, None] == np.arange(1, len(self.categories_))).astype(np.float32)

    def _epredict(self, model, z):
        if not self.discrete_treatment:
            return _pred_vector(model, z)[:, None]
        p = np.asarray(model.predict_proba(z), dtype=float)
        classes = list(model.classes_)
        if (p.shape != (z.shape[0], len(classes)) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
            raise ValueError('Invalid nuisance treatment probabilities.')
        try:
            return p[:, [classes.index(v) for v in self.categories_[1:]]]
        except ValueError as e:
            raise ValueError('Treatment model is missing a category.') from e

    def _folds(self, x, t, groups):
        n = len(t)
        if isinstance(self.cv, (int, np.integer)) and not isinstance(self.cv, bool):
            if self.cv < 2:
                raise ValueError('cv must be at least 2.')
            if groups is not None:
                split = (StratifiedGroupKFold(self.cv, shuffle=True, random_state=self.random_state)
                         if self.discrete_treatment else GroupKFold(self.cv))
            else:
                cls = StratifiedKFold if self.discrete_treatment else KFold
                split = cls(self.cv, shuffle=True, random_state=self.random_state)
            folds = list(split.split(x, t, groups))
        elif hasattr(self.cv, 'split'):
            folds = list(self.cv.split(x, t, groups))
        else:
            folds = list(self.cv)
        seen = np.zeros(n, dtype=np.int32)
        checked = []
        for tr, va in folds:
            tr, va = np.asarray(tr), np.asarray(va)
            for a in (tr, va):
                if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a)
                        or a.min() < 0 or a.max() >= n or len(np.unique(a)) != len(a)):
                    raise ValueError('Invalid or duplicate fold indices.')
            if np.intersect1d(tr, va).size:
                raise ValueError('Train/validation row leakage.')
            if groups is not None and set(groups[tr]) & set(groups[va]):
                raise ValueError('Group leakage between folds.')
            if self.discrete_treatment and set(t[tr]) != set(self.categories_):
                raise ValueError('Each training fold must contain every treatment.')
            seen[va] += 1
            checked.append((tr, va))
        if not checked or not np.all(seen == 1):
            raise ValueError('Each row must be held out exactly once.')
        return checked

    @staticmethod
    def _aligned(*values):
        indices = [v.index for v in values if isinstance(v, (pd.Series, pd.DataFrame))]
        if indices and any(not idx.equals(indices[0]) for idx in indices[1:]):
            raise ValueError('Pandas inputs must have identical row indices and order.')

    def _x(self, X, n=None):
        if X is None:
            if self.n_features_in_:
                raise ValueError('X is required for this fitted model.')
            return np.ones((1 if n is None else n, 1), dtype=np.float32)
        x = _numeric(X, 'X')
        if self.n_features_in_ == 0 or x.shape[1] != self.n_features_in_:
            raise ValueError('Wrong X width; use X=None for a constant-effect model.')
        if self.x_columns_ is not None and isinstance(X, pd.DataFrame) and list(X.columns) != self.x_columns_:
            raise ValueError('X column names/order changed.')
        return x

    def fit(self, Y, T, *, X=None, W=None, groups=None):
        self.is_fitted_ = False
        self.config_ = self.final_config or FinalStageConfig()
        self.config_.validate()
        if (not np.isfinite(self.ridge) or self.ridge < 0
                or not np.isfinite(self.identification_tol) or self.identification_tol <= 0
                or isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int)
                or self.batch_size < 1 or not isinstance(self.n_jobs, int) or self.n_jobs < 1):
            raise ValueError('Invalid ridge, identification_tol, batch_size, or n_jobs.')
        self._aligned(Y, T, X, W, groups)
        y = _vector(Y, 'Y').astype(float)
        t = _vector(T, 'T')
        n = len(y)
        if not np.isfinite(y).all() or len(t) != n or pd.isna(t).any():
            raise ValueError('Invalid Y/T values or row count.')
        self.n_features_in_ = 0 if X is None else _numeric(X, 'X').shape[1]
        self.x_columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.w_columns_ = list(W.columns) if isinstance(W, pd.DataFrame) else None
        x = self._x(X, n)
        w = None if W is None else _numeric(W, 'W')
        self.n_w_features_ = 0 if w is None else w.shape[1]
        if x.shape[0] != n or (w is not None and w.shape[0] != n):
            raise ValueError('X/W row count differs from Y.')
        z = _combine(x, w)
        if groups is not None:
            if np.asarray(groups).ndim != 1:
                raise ValueError('groups must be a one-dimensional vector.')
            groups = _vector(groups, 'groups')
            if len(groups) != n or pd.isna(groups).any():
                raise ValueError('Invalid groups.')
        if self.discrete_treatment:
            observed = list(pd.unique(t))
            if self.categories is None:
                warnings.warn('Pass categories explicitly to fix the reference treatment.', UserWarning)
            self.categories_ = list(self.categories) if self.categories is not None else sorted(observed, key=str)
            if (len(self.categories_) < 2 or len(set(self.categories_)) != len(self.categories_)
                    or set(observed) != set(self.categories_)):
                raise ValueError('categories must be unique, all observed, and include all treatments.')
            self._category_index_ = {v: i for i, v in enumerate(self.categories_)}
        else:
            self.categories_ = None
            if not np.isfinite(t.astype(float)).all():
                raise ValueError('Continuous T must be finite.')
        q = len(self.categories_) - 1 if self.discrete_treatment else 1
        self.n_treatments_ = q
        folds = self._folds(z, t, groups)
        parent = Path(self.work_dir) if self.work_dir else None
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        self.fold_audit_ = []
        with tempfile.TemporaryDirectory(prefix='causal_dml_', dir=parent) as td:
            r = np.memmap(Path(td) / 'residuals.f32', dtype='float32', mode='w+', shape=(n, q))
            yr = np.empty(n, dtype=float)
            for k, (tr, va) in enumerate(folds):
                my, mt = self._nuisance(), self._nuisance(True)
                ztr, zva = z[tr], z[va]
                my.fit(ztr, y[tr])
                yr[va] = y[va] - _pred_vector(my, zva)
                del my
                mt.fit(ztr, t[tr])
                r[va] = self._design(t[va]) - self._epredict(mt, zva)
                self.fold_audit_.append(dict(fold=k, train_rows=len(tr), validation_rows=len(va), shared_groups=0))
                del mt, ztr, zva
            gram, rhs = np.zeros((q, q)), np.zeros(q)
            for start in range(0, n, self.batch_size):
                rb = np.asarray(r[start:start+self.batch_size], dtype=float)
                gram += rb.T @ rb
                rhs += rb.T @ yr[start:start+self.batch_size]
            gram /= n
            rhs /= n
            eig = np.linalg.eigvalsh(gram)
            threshold = self.identification_tol * max(1., float(eig[-1]))
            self.identification_ = dict(eigenvalues=eig.tolist(), min_eigenvalue=float(eig[0]),
                                        condition_number=float(eig[-1] / eig[0]) if eig[0] > 0 else None,
                                        weak=bool(eig[0] < threshold), threshold=threshold)
            if self.identification_['weak']:
                warnings.warn('Weak global residual treatment identification. Ridge does not fix overlap.', UserWarning)
            self.constant_effect_ = np.linalg.solve(gram + self.ridge * np.eye(q), rhs)
            self.models_ = [None] * q
            fitted = np.asarray(r @ self.constant_effect_, dtype=float)
            loss = float(np.mean((yr - fitted) ** 2))
            self.loss_history_ = [loss]
            self.update_history_ = []
            self.final_stage_constant_ = self.n_features_in_ == 0 or not _has_variation(x)
            if not self.final_stage_constant_:
                # Construct histogram bins once and reuse them across coordinates.
                ds = lgb.Dataset(x, label=np.zeros(n), free_raw_data=True,
                                 params={'max_bin': self.config_.max_bin, 'feature_pre_filter': False,
                                         'num_threads': self.n_jobs, 'verbosity': -1})
                ds.construct()
                for sweep in range(self.config_.backfit_rounds):
                    before = loss
                    for j in range(q):
                        rj = np.asarray(r[:, j], dtype=float)
                        old = self._coordinate(x, j)
                        # New coordinate is baseline_j + f_j(X), replacing its old tree model.
                        partial = yr - fitted + rj * (old - self.constant_effect_[j])
                        ds.set_label(partial)

                        def objective(pred, data):
                            return orthogonal_derivatives(pred, data.get_label(), rj)

                        cfg = self.config_
                        params = dict(objective=objective, metric='None', boosting_type='gbdt',
                                      boost_from_average=False, learning_rate=cfg.learning_rate,
                                      num_leaves=cfg.num_leaves, max_depth=cfg.max_depth,
                                      min_data_in_leaf=cfg.min_child_samples,
                                      min_sum_hessian_in_leaf=cfg.min_child_weight,
                                      bagging_fraction=cfg.subsample, bagging_freq=1,
                                      feature_fraction=cfg.colsample_bytree, extra_trees=cfg.extra_trees,
                                      lambda_l2=cfg.reg_lambda, lambda_l1=cfg.reg_alpha,
                                      max_bin=cfg.max_bin, feature_pre_filter=False,
                                      force_col_wise=True, num_threads=self.n_jobs, verbosity=-1,
                                      seed=self.random_state + j, deterministic=True)
                        model = lgb.train(params, ds, num_boost_round=cfg.n_estimators)
                        pred = self.constant_effect_[j] + model.predict(x, num_threads=self.n_jobs)
                        candidate = fitted + rj * (pred - old)
                        new_loss = float(np.mean((yr - candidate) ** 2))
                        accepted = np.isfinite(new_loss) and new_loss < loss
                        self.update_history_.append(dict(sweep=sweep, coordinate=j,
                                                         candidate_loss=new_loss, accepted=bool(accepted)))
                        if accepted:
                            self.models_[j], fitted, loss = model, candidate, new_loss
                    self.loss_history_.append(loss)
                    if before - loss <= cfg.tolerance * max(before, 1e-12):
                        break
                del ds
            self.training_residual_loss_ = loss
            r.flush()
            del r
        # Full-data nuisances are for independent validation/test diagnostics.
        # Cross-fit models and residuals are not retained.
        self.model_y_ = self._nuisance()
        self.model_y_.fit(z, y)
        self.model_t_ = self._nuisance(True)
        self.model_t_.fit(z, t)
        self.is_fitted_ = True
        return self

    def _check(self):
        check_is_fitted(self, 'models_')
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError('fit must complete successfully before prediction.')

    def _coordinate(self, x, j):
        p = np.full(x.shape[0], self.constant_effect_[j])
        if self.models_[j] is not None:
            p += self.models_[j].predict(x, num_threads=self.n_jobs)
        return p

    def const_marginal_effect(self, X=None):
        """(n,q) contrasts versus categories_[0]; continuous: (n,1) dose slope."""
        self._check()
        x = self._x(X)
        return np.column_stack([self._coordinate(x, j) for j in range(self.n_treatments_)])

    @staticmethod
    def _endpoints(v, default, n):
        a = np.asarray(default if v is None else v)
        if a.ndim > 1:
            raise ValueError('T0/T1 must be scalars or row-wise vectors.')
        try:
            return np.broadcast_to(a, (n,))
        except ValueError as e:
            raise ValueError('T0/T1 lengths must equal prediction rows.') from e

    def effect(self, X=None, *, T0=None, T1=None):
        """Effect of changing T0 to T1. Selected contrasts avoid an n-by-q output."""
        self._check()
        x = self._x(X)
        n = x.shape[0]
        if not self.discrete_treatment:
            lo = self._endpoints(T0, 0., n).astype(float)
            hi = self._endpoints(T1, 1., n).astype(float)
            if not np.isfinite(lo).all() or not np.isfinite(hi).all():
                raise ValueError('T0/T1 must be finite.')
            return self._coordinate(x, 0) * (hi - lo)
        try:
            lo = np.array([self._category_index_[v] for v in self._endpoints(T0, self.categories_[0], n)])
            hi = np.array([self._category_index_[v] for v in self._endpoints(T1, self.categories_[1], n)])
        except (KeyError, TypeError) as e:
            raise ValueError('Unknown T0/T1 treatment.') from e
        result = np.zeros(n)
        for arm in np.union1d(lo, hi):
            if arm:
                mask = (lo == arm) | (hi == arm)
                result[mask] += self._coordinate(x[mask], int(arm)-1) * ((hi[mask] == arm).astype(int) - (lo[mask] == arm))
        return result

    def iter_effect(self, X, *, T0=None, T1=None, batch_size=None):
        """Yield bounded-memory contrast batches; row-wise endpoints supported."""
        self._check()
        x = self._x(X)
        size = self.batch_size if batch_size is None else batch_size
        if not isinstance(size, int) or size < 1:
            raise ValueError('batch_size must be positive.')
        lo = self._endpoints(T0, self.categories_[0] if self.discrete_treatment else 0., x.shape[0])
        hi = self._endpoints(T1, self.categories_[1] if self.discrete_treatment else 1., x.shape[0])
        # x already passed the DataFrame schema check above.
        for start in range(0, x.shape[0], size):
            end = start + size
            yield self.effect(x[start:end], T0=lo[start:end], T1=hi[start:end])

    def ate(self, X=None, *, T0=None, T1=None):
        """Empirical mean CATE over supplied X; no standard error or population weighting."""
        if X is None:
            return float(self.effect(None, T0=T0, T1=T1).mean())
        total, n = 0., 0
        for e in self.iter_effect(X, T0=T0, T1=T1):
            total += float(e.sum())
            n += len(e)
        if not n:
            raise ValueError('Cannot average an empty population.')
        return total / n

    def residual_diagnostics(self, Y, T, *, X=None, W=None):
        """Independent-data residual fit, not evidence of causal identification."""
        self._check()
        self._aligned(Y, T, X, W)
        y, t = _vector(Y, 'Y').astype(float), _vector(T, 'T')
        x = self._x(X, len(y))
        w = None if W is None else _numeric(W, 'W')
        if (len(y) != len(t) or x.shape[0] != len(y) or not np.isfinite(y).all()
                or (w is not None and w.shape[0] != len(y))
                or (0 if w is None else w.shape[1]) != self.n_w_features_):
            raise ValueError('Diagnostic row counts/feature widths do not match.')
        if self.w_columns_ is not None and isinstance(W, pd.DataFrame) and list(W.columns) != self.w_columns_:
            raise ValueError('W column names/order changed.')
        sse, baseline_sse, null_sse = 0., 0., 0.
        for start in range(0, len(y), self.batch_size):
            sl = slice(start, start+self.batch_size)
            xb = x[sl]
            z = _combine(xb, None if w is None else w[sl])
            yr = y[sl] - _pred_vector(self.model_y_, z)
            r = self._design(t[sl]) - self._epredict(self.model_t_, z)
            pred = np.zeros(len(yr))
            for j in range(self.n_treatments_):
                pred += r[:, j] * self._coordinate(xb, j)
            sse += float(np.sum((yr - pred) ** 2))
            baseline_sse += float(np.sum((yr - r @ self.constant_effect_) ** 2))
            null_sse += float(np.sum(yr ** 2))
        return dict(cate_residual_loss=sse / len(y), constant_residual_loss=baseline_sse / len(y),
                    null_residual_loss=null_sse / len(y),
                    r_score_vs_constant=1 - sse / baseline_sse if baseline_sse > 0 else None,
                    note='Use independent validation/test rows. This is not causal ground truth.')

    def score(self, Y, T, *, X=None, W=None):
        """Negative residual MSE; larger is better."""
        return -self.residual_diagnostics(Y, T, X=X, W=W)['cate_residual_loss']


class LightweightDRLearner(CustomCausalDML):
    """Categorical DR CATE learner and independent policy evaluator.

    A shared S-learner estimates E[Y|X,W,T]. Cross-fitted AIPW contrasts are
    regressed on X using bounded LightGBM trees, one arm at a time. All code
    remains in this standalone module. Scalar continuous T uses CustomCausalDML.
    """
    def __init__(self, *, categories=None, cv=3, model_y=None, model_t=None,
                 final_config=None, propensity_floor=1e-3, known_propensities=None,
                 random_state=2026, n_jobs=2, batch_size=20000, work_dir=None):
        super().__init__(categories=categories, cv=cv, model_y=model_y, model_t=model_t,
                         final_config=final_config, random_state=random_state,
                         n_jobs=n_jobs, batch_size=batch_size, work_dir=work_dir)
        self.propensity_floor = propensity_floor
        self.known_propensities = known_propensities

    def _ids(self, t):
        try:
            return np.array([self._category_index_[v] for v in t], dtype=int)
        except (KeyError, TypeError) as e:
            raise ValueError('Unknown treatment label.') from e

    def _augment(self, z, ids):
        n, q = z.shape[0], len(self.categories_)-1
        mask = ids > 0
        code = sparse.csr_matrix((np.ones(mask.sum(), dtype=np.float32),
                                 (np.flatnonzero(mask), ids[mask]-1)), shape=(n, q))
        return sparse.hstack([sparse.csr_matrix(z), code], format='csr', dtype=np.float32)

    def _probabilities(self, model, z):
        if self.known_propensities_ is not None:
            return np.broadcast_to(self.known_propensities_, (z.shape[0], len(self.categories_)))
        p = np.asarray(model.predict_proba(z), dtype=float)
        classes = list(model.classes_)
        try:
            p = p[:, [classes.index(v) for v in self.categories_]]
        except (ValueError, IndexError) as e:
            raise ValueError('Missing propensity class.') from e
        if (p.shape != (z.shape[0], len(self.categories_)) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise ValueError('Invalid propensity probabilities.')
        return p

    def _pseudo(self, y, t_ids, z, my, mt):
        p = self._probabilities(mt, z)
        observed_mu = _pred_vector(my, self._augment(z, t_ids))
        residual = y-observed_mu
        for arm in range(len(self.categories_)):
            mu = _pred_vector(my, self._augment(z, np.full(len(y), arm)))
            yield arm, mu+(t_ids == arm)*residual/np.maximum(p[:, arm], self.propensity_floor), p[:, arm]

    def fit(self, Y, T, *, X=None, W=None, groups=None):
        self.is_fitted_ = False
        self._aligned(Y, T, X, W, groups)
        self.config_ = self.final_config or FinalStageConfig()
        self.config_.validate()
        if (not np.isfinite(self.propensity_floor) or not 0 < self.propensity_floor < .5
                or not isinstance(self.batch_size, int) or self.batch_size < 1
                or not isinstance(self.n_jobs, int) or self.n_jobs < 1):
            raise ValueError('Invalid propensity_floor, batch_size or n_jobs.')
        y, t = _vector(Y, 'Y').astype(float), _vector(T, 'T')
        n = len(y)
        if len(t) != n or not np.isfinite(y).all() or pd.isna(t).any():
            raise ValueError('Invalid Y/T values or row counts.')
        observed = list(pd.unique(t))
        if self.categories is None:
            warnings.warn('Pass categories explicitly to fix the reference.', UserWarning)
        self.categories_ = list(self.categories) if self.categories is not None else sorted(observed, key=str)
        if (len(self.categories_) < 2 or len(set(self.categories_)) != len(self.categories_)
                or set(observed) != set(self.categories_)):
            raise ValueError('categories must be unique and match all observed treatments.')
        self._category_index_ = {v: i for i, v in enumerate(self.categories_)}
        ids = self._ids(t)
        self.n_treatments_ = len(self.categories_)-1
        self.n_features_in_ = 0 if X is None else _numeric(X, 'X').shape[1]
        self.x_columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.w_columns_ = list(W.columns) if isinstance(W, pd.DataFrame) else None
        x = self._x(X, n)
        w = None if W is None else _numeric(W, 'W')
        self.n_w_features_ = 0 if w is None else w.shape[1]
        if x.shape[0] != n or (w is not None and w.shape[0] != n):
            raise ValueError('X/W row count mismatch.')
        z = _combine(x, w)
        if groups is not None:
            groups = np.asarray(groups)
            if groups.ndim != 1 or len(groups) != n or pd.isna(groups).any():
                raise ValueError('groups must be a nonmissing row-wise vector.')
        self.training_groups_ = None if groups is None else frozenset(groups.tolist())
        self.calibration_groups_ = frozenset()
        self.calibrators_ = [None]*self.n_treatments_
        if self.known_propensities is None:
            self.known_propensities_ = None
        else:
            p = np.asarray(self.known_propensities, dtype=float)
            if p.shape != (len(self.categories_),) or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1):
                raise ValueError('known_propensities must be a positive probability vector in categories order.')
            self.known_propensities_ = p
        folds = self._folds(z, t, groups)
        parent = Path(self.work_dir) if self.work_dir else None
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        self.fold_audit_ = []
        clip_counts = np.zeros(len(self.categories_), dtype=int)
        with tempfile.TemporaryDirectory(prefix='dr_dml_', dir=parent) as td:
            targets = np.memmap(Path(td)/'contrasts.f32', dtype='float32', mode='w+', shape=(n, self.n_treatments_))
            for k, (tr, va) in enumerate(folds):
                my, mt = self._nuisance(), None if self.known_propensities_ is not None else self._nuisance(True)
                my.fit(self._augment(z[tr], ids[tr]), y[tr])
                if mt is not None:
                    mt.fit(z[tr], t[tr])
                for start in range(0, len(va), self.batch_size):
                    rows = va[start:start+self.batch_size]
                    reference = None
                    for arm, phi, p in self._pseudo(y[rows], ids[rows], z[rows], my, mt):
                        clip_counts[arm] += int((p < self.propensity_floor).sum())
                        if arm == 0:
                            reference = phi
                        else:
                            targets[rows, arm-1] = phi-reference
                self.fold_audit_.append(dict(fold=k, train_rows=len(tr), validation_rows=len(va), shared_groups=0))
                del my, mt
            self.constant_effect_ = np.asarray(targets.mean(axis=0, dtype=float))
            self.models_ = [None]*self.n_treatments_
            self.final_stage_constant_ = self.n_features_in_ == 0 or not _has_variation(x)
            self.training_pseudo_losses_ = []
            if not self.final_stage_constant_:
                ds = lgb.Dataset(x, label=np.zeros(n), free_raw_data=True,
                                 params={'max_bin': self.config_.max_bin, 'feature_pre_filter': False,
                                         'num_threads': self.n_jobs, 'verbosity': -1})
                ds.construct()
                cfg = self.config_
                for j in range(self.n_treatments_):
                    target = np.asarray(targets[:, j], dtype=float)-self.constant_effect_[j]
                    ds.set_label(target)
                    params = dict(objective='regression', metric='None', boost_from_average=False,
                        num_leaves=cfg.num_leaves, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                        min_data_in_leaf=cfg.min_child_samples, min_sum_hessian_in_leaf=cfg.min_child_weight,
                        bagging_fraction=cfg.subsample, bagging_freq=1, feature_fraction=cfg.colsample_bytree,
                        extra_trees=cfg.extra_trees, lambda_l2=cfg.reg_lambda, lambda_l1=cfg.reg_alpha,
                        max_bin=cfg.max_bin, feature_pre_filter=False, force_col_wise=True,
                        num_threads=self.n_jobs, seed=self.random_state+j, verbosity=-1, deterministic=True)
                    model = lgb.train(params, ds, num_boost_round=cfg.n_estimators)
                    pred = model.predict(x, num_threads=self.n_jobs)
                    loss = float(np.mean((target-pred)**2))
                    if loss < np.mean(target**2):
                        self.models_[j] = model
                    self.training_pseudo_losses_.append(loss)
                del ds
            targets.flush()
            del targets
        self.overlap_ = dict(propensity_floor=self.propensity_floor,
                             clipped_fraction_by_arm=(clip_counts/n).tolist(),
                             treatment_counts=pd.Series(t).value_counts().to_dict())
        self.model_y_ = self._nuisance()
        self.model_y_.fit(self._augment(z, ids), y)
        self.model_t_ = None if self.known_propensities_ is not None else self._nuisance(True)
        if self.model_t_ is not None:
            self.model_t_.fit(z, t)
        self.is_fitted_ = True
        return self

    def _coordinate(self, x, j):
        p = super()._coordinate(x, j)
        if self.calibrators_[j] is not None:
            p = self.calibrators_[j].predict(p)
        return p

    def _data(self, X, W, n=None):
        if n is None and X is None and W is not None:
            n = W.shape[0] if hasattr(W, 'shape') else len(W)
        x = self._x(X, n)
        w = None if W is None else _numeric(W, 'W')
        if ((0 if w is None else w.shape[1]) != self.n_w_features_
                or (w is not None and w.shape[0] != x.shape[0])):
            raise ValueError('W schema/rows differ from training.')
        if self.w_columns_ is not None and isinstance(W, pd.DataFrame) and list(W.columns) != self.w_columns_:
            raise ValueError('W column order changed.')
        return x, w

    def _independent(self, groups, n, *, calibration=False):
        if groups is None:
            if self.training_groups_ is not None:
                raise ValueError('Supply groups so held-out separation can be verified.')
            return None
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != n or pd.isna(groups).any():
            raise ValueError('Invalid evaluation groups.')
        blocked = (self.training_groups_ or frozenset()) | (frozenset() if calibration else self.calibration_groups_)
        if set(groups) & blocked:
            raise ValueError('Evaluation groups overlap fitting/calibration groups.')
        return groups

    def iter_dr_scores(self, Y, T, *, X=None, W=None):
        """Yield (start, n_batch-by-K AIPW potential-outcome scores, probabilities).

        These are noisy scores, not potential-outcome predictions for individuals.
        Caller must use independent data for validation/inference.
        """
        self._check()
        self._aligned(Y, T, X, W)
        y, t = _vector(Y, 'Y').astype(float), _vector(T, 'T')
        x, w = self._data(X, W, len(y))
        if len(t) != len(y) or x.shape[0] != len(y) or not np.isfinite(y).all():
            raise ValueError('Invalid score inputs.')
        ids = self._ids(t)
        for start in range(0, len(y), self.batch_size):
            sl = slice(start, start+self.batch_size)
            z = _combine(x[sl], None if w is None else w[sl])
            vals, probs = [], []
            for _, phi, p in self._pseudo(y[sl], ids[sl], z, self.model_y_, self.model_t_):
                vals.append(phi)
                probs.append(p)
            yield start, np.column_stack(vals), np.column_stack(probs)

    def residual_diagnostics(self, *args, **kwargs):
        raise NotImplementedError('Use dr_diagnostics for a DR learner; R-loss needs different nuisances.')

    def dr_diagnostics(self, Y, T, *, X=None, W=None, groups=None):
        self._check()
        n = len(_vector(Y, 'Y'))
        self._independent(groups, n)
        x, _ = self._data(X, W, n)
        loss, base = 0., 0.
        for start, phi, p in self.iter_dr_scores(Y, T, X=X, W=W):
            for j in range(self.n_treatments_):
                target = phi[:, j+1]-phi[:, 0]
                pred = self._coordinate(x[start:start+len(phi)], j)
                loss += float(np.sum((target-pred)**2))
                base += float(np.sum((target-self.constant_effect_[j])**2))
        return dict(dr_pseudo_mse=loss/(n*self.n_treatments_),
                    constant_pseudo_mse=base/(n*self.n_treatments_),
                    score_vs_constant=1-loss/base if base > 0 else None,
                    note='Independent pseudo-outcome loss; noisy and not causal ground truth.')

    def score(self, Y, T, *, X=None, W=None, groups=None):
        return -self.dr_diagnostics(Y, T, X=X, W=W, groups=groups)['dr_pseudo_mse']

    def fit_calibration(self, Y, T, *, X=None, W=None, groups=None):
        """Optional monotone CATE calibration on an independent dedicated split."""
        from sklearn.isotonic import IsotonicRegression
        self._check()
        n = len(_vector(Y, 'Y'))
        g = self._independent(groups, n, calibration=True)
        if any(c is not None for c in self.calibrators_):
            raise ValueError('Calibration already fitted; refit the estimator to replace it.')
        x, _ = self._data(X, W, n)
        if n < 100:
            raise ValueError('At least 100 independent calibration rows required.')
        # One contrast at a time; O(n) storage rather than full n-by-K in RAM.
        for j in range(self.n_treatments_):
            pred, target = [], []
            for start, phi, _ in self.iter_dr_scores(Y, T, X=X, W=W):
                pred.append(super()._coordinate(x[start:start+len(phi)], j))
                target.append(phi[:, j+1]-phi[:, 0])
            pred, target = np.concatenate(pred), np.concatenate(target)
            if len(np.unique(pred)) > 1:
                self.calibrators_[j] = IsotonicRegression(out_of_bounds='clip').fit(pred, target)
        self.calibration_groups_ = frozenset() if g is None else frozenset(g.tolist())
        return self

    def recommend(self, X=None, *, W=None, costs=None, min_propensity=.01,
                  min_gain=0., eligible=None, cost_multiplier=1.):
        """Return a supported best action per row. Costs are known same-scale amounts.

        Row-specific p support uses W, so policy may depend on W as well as X.
        Reference is used when no supported improvement is available. This is
        a plug-in policy, not a guarantee that its true value is positive.
        """
        self._check()
        if not 0 < min_propensity < 1 or not np.isfinite(min_gain) or min_gain < 0 or not np.isfinite(cost_multiplier) or cost_multiplier < 0:
            raise ValueError('Invalid policy thresholds.')
        x, w = self._data(X, W)
        n, k = x.shape[0], len(self.categories_)
        c = self._costs(costs, n, k)
        if eligible is not None and np.asarray(eligible).shape != (n, k):
            raise ValueError('eligible must be n-by-K.')
        action_ids, gains, support = np.zeros(n, dtype=int), np.zeros(n), np.zeros(n, dtype=bool)
        for start in range(0, n, self.batch_size):
            sl = slice(start, start+self.batch_size)
            xb = x[sl]
            p = self._probabilities(self.model_t_, _combine(xb, None if w is None else w[sl]))
            ok = p >= min_propensity
            if eligible is not None:
                ok &= np.asarray(eligible, dtype=bool)[sl]
            ok &= ok[:, :1]  # reference must also have support
            values = np.zeros((len(p), k))
            for j in range(k-1):
                values[:, j+1] = self._coordinate(xb, j)
            values -= cost_multiplier*(c[sl]-c[sl, :1])
            values[~ok] = -np.inf
            values[:, 0] = 0.  # reference fallback can still be unsupported; report that explicitly
            choice = values.argmax(axis=1)
            gain = values[np.arange(len(p)), choice]
            choice[gain <= min_gain] = 0
            action_ids[sl] = choice
            gains[sl] = values[np.arange(len(p)), choice]
            support[sl] = ok[np.arange(len(p)), choice]
        return dict(action=np.array(self.categories_, dtype=object)[action_ids], action_index=action_ids,
                    predicted_net_gain=gains, supported=support)

    @staticmethod
    def _costs(costs, n, k):
        a = np.zeros(k) if costs is None else np.asarray(costs, dtype=float)
        if a.shape not in [(k,), (n, k)] or not np.isfinite(a).all():
            raise ValueError('costs must be finite K or n-by-K known costs in outcome units.')
        return np.broadcast_to(a, (n, k))

    def evaluate_policy(self, Y, T, *, X=None, W=None, policy=None, baseline=None,
                        groups=None, costs=None, min_propensity=.01, alpha=.05):
        """DR value difference for a fixed independent policy; clustered Wald interval.

        Reports the full population estimate with overlap failures flagged. It
        does not silently trim to a policy-dependent subpopulation. Interval is
        conditional on learned nuisances/policy, under asymptotic assumptions;
        no valid pointwise CATE inference is claimed.
        """
        from scipy.stats import t as student_t
        self._check()
        y = _vector(Y, 'Y').astype(float)
        n, k = len(y), len(self.categories_)
        g = self._independent(groups, n)
        if not 0 < alpha < 1 or not 0 < min_propensity < 1:
            raise ValueError('Invalid alpha or min_propensity.')
        pi = self._ids(self._endpoints(policy, self.categories_[0], n))
        b = self._ids(self._endpoints(baseline, self.categories_[0], n))
        c = self._costs(costs, n, k)
        score = np.empty(n)
        baseline_score = np.empty(n)
        weights_pi, weights_b = np.zeros(n), np.zeros(n)
        t_ids = self._ids(_vector(T, 'T'))
        failures, clipped = 0, 0
        for start, phi, p in self.iter_dr_scores(Y, T, X=X, W=W):
            sl = slice(start, start+len(phi))
            ii = np.arange(len(phi))
            pa, pb = p[ii, pi[sl]], p[ii, b[sl]]
            failures += int(((pa < min_propensity) | (pb < min_propensity)).sum())
            clipped += int(((pa < self.propensity_floor) | (pb < self.propensity_floor)).sum())
            score[sl] = phi[ii, pi[sl]]-phi[ii, b[sl]]-c[sl][ii, pi[sl]]+c[sl][ii, b[sl]]
            baseline_score[sl] = phi[ii, b[sl]]-c[sl][ii, b[sl]]
            weights_pi[sl] = (t_ids[sl] == pi[sl])/np.maximum(pa, self.propensity_floor)
            weights_b[sl] = (t_ids[sl] == b[sl])/np.maximum(pb, self.propensity_floor)
        mean = float(score.mean())
        baseline_mean = float(baseline_score.mean())
        if g is None:
            se = float(score.std(ddof=1)/np.sqrt(n)) if n > 1 else None
            count = n
        else:
            centered = pd.Series(score-mean).groupby(g, sort=False).sum().to_numpy()
            count = len(centered)
            se = float(np.sqrt(count/(count-1)*np.sum(centered**2))/n) if count > 1 else None
        quantile = float(student_t.ppf(1-alpha/2, count-1)) if count > 1 else None
        ess = lambda v: float(v.sum()**2/(v@v)) if v@v > 0 else 0.
        return dict(mean_gain=mean, baseline_mean_value=baseline_mean,
                    policy_mean_value=baseline_mean+mean,
                    relative_gain_pct=100*mean/baseline_mean if baseline_mean > 0 else None,
                    relative_gain_note='Ratio of mean values, not a mean of row percentages. '
                                       'Undefined for nonpositive baseline; the gain CI is not a percentage CI.',
                    standard_error=se,
                    ci_low=mean-quantile*se if se is not None else None,
                    ci_high=mean+quantile*se if se is not None else None,
                    alpha=alpha, rows=n, independent_units=count,
                    policy_ess=ess(weights_pi), baseline_ess=ess(weights_b),
                    unsupported_fraction=failures/n, clipped_fraction=clipped/n,
                    overlap_pass=bool(failures == 0 and clipped == 0),
                    note='Fixed-policy full-population DR contrast; clustered when groups supplied. '
                         'Clipping can bias the propensity-correct branch. CI does not cover hidden confounding, '
                         'support violations, nuisance error outside rate assumptions, or policy selection on these rows.')
