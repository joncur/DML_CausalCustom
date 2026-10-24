"""Opt-in controlled final-stage benchmark; no user data, no causal accuracy claim.

Each backend runs in a fresh child process. RF is a conventional weighted
residual-loss regressor, NOT EconML CausalForestDML. RSS includes data generation.
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np


def worker(backend, n):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(n, 8)).astype('float32')
    r = rng.binomial(1, .5, n)-.5
    tau = 1 + .5*x[:, 0]
    yr = r*tau+rng.normal(0, 1, n)
    target, weight = yr/r, r*r  # exactly +/- .5; no small-denominator issue in this fixture
    if backend == 'lightgbm':
        model = LGBMRegressor(n_estimators=160, learning_rate=.05, num_leaves=15,
            max_depth=5, min_child_samples=200, subsample=.7, subsample_freq=1,
            colsample_bytree=.8, extra_trees=True, reg_lambda=5, max_bin=63,
            n_jobs=2, verbosity=-1, random_state=2026)
    else:
        model = RandomForestRegressor(n_estimators=160, min_samples_leaf=5, max_features=.8,
                                      n_jobs=2, random_state=2026)
    start = time.perf_counter()
    model.fit(x, target, sample_weight=weight)
    seconds = time.perf_counter()-start
    if backend == 'lightgbm':
        model_bytes = len(model.booster_.model_to_string().encode())
        nodes = sum(t['num_leaves']*2-1 for t in model.booster_.dump_model()['tree_info'])
    else:
        model_bytes = sum(t.tree_.__getstate__()['nodes'].nbytes+t.tree_.value.nbytes for t in model.estimators_)
        nodes = sum(t.tree_.node_count for t in model.estimators_)
    print(json.dumps(dict(backend=backend, rows=n, seconds=seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_bytes=model_bytes, nodes=nodes)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=50000)
    p.add_argument('--worker', choices=['lightgbm', 'random_forest'])
    args = p.parse_args()
    if args.worker:
        worker(args.worker, args.rows)
    else:
        for backend in ('lightgbm', 'random_forest'):
            subprocess.run([sys.executable, __file__, '--rows', str(args.rows), '--worker', backend], check=True)
