import json
import os

import numpy as np
import pandas as pd


def sample_configs(space, n_trials, seed):
    rng = np.random.default_rng(seed)

    def draw(spec, first):
        if isinstance(spec, list):
            return spec[0] if first else spec[int(rng.integers(len(spec)))]
        kind, lo, hi = spec
        if kind == "loguniform":
            return float(f"{np.sqrt(lo * hi) if first else np.exp(rng.uniform(np.log(lo), np.log(hi))):.3g}")
        return float(f"{(lo + hi) / 2 if first else rng.uniform(lo, hi):.3g}")

    configs, seen, attempts = [], set(), 0
    while len(configs) < n_trials and attempts < n_trials * 20:
        cfg = {k: draw(v, not configs and not attempts) for k, v in space.items()}
        key = json.dumps(cfg, sort_keys=True)
        attempts += 1
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def random_search(model_cls, space, evaluator, n_trials, seed=0, fixed=None, metric="ndcg@10", cache_path=None):
    done = {}
    if cache_path and os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            rec = json.loads(line)
            done[json.dumps(rec["params"], sort_keys=True)] = rec
    rows = []
    for n, cfg in enumerate(sample_configs(space, n_trials, seed), 1):
        key = json.dumps(cfg, sort_keys=True)
        if key not in done:
            done[key] = {"params": cfg, "metrics": evaluator.fit_and_evaluate(model_cls(**(fixed or {}), **cfg))}
            if cache_path:
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(done[key]) + "\n")
        m = done[key]["metrics"]
        print(f"[{model_cls.name} {n}/{n_trials}] {metric}={m[metric]:.4f} fit={m.get('fit_time', 0):.0f}s {cfg}")
        rows.append({**{f"param_{k}": v for k, v in cfg.items()}, **m})
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def best_params(results):
    row = results.iloc[0]
    return {c[6:]: (v.item() if isinstance(v, np.generic) else v) for c, v in row.items() if c.startswith("param_")}
