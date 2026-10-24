"""Opt-in controlled full DR fit benchmark; not production data or accuracy evidence."""
import argparse
import json
import resource
import time
import numpy as np
from causal_dml import LightweightDRLearner, FinalStageConfig


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=50000)
    args = p.parse_args()
    rng = np.random.default_rng(7)
    n = args.rows
    x = rng.normal(size=(n,5)).astype('float32')
    w = rng.normal(size=(n,2)).astype('float32')
    t = rng.binomial(1,.5,n)
    y = w[:,0]+t*(1+.4*x[:,0])+rng.normal(size=n)
    start = time.perf_counter()
    m = LightweightDRLearner(categories=[0,1], known_propensities=[.5,.5],
        final_config=FinalStageConfig(n_estimators=80,num_leaves=7,max_depth=3),cv=3)
    m.fit(y,t,X=x,W=w)
    print(json.dumps(dict(rows=n,features=7,arms=2,seconds=time.perf_counter()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        retained_tree_model_bytes=sum(len(a.model_to_string().encode()) for a in m.models_ if a is not None),
        note='Controlled full DR fit with known randomized propensities; not production accuracy evidence.')))
