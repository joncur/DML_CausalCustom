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
