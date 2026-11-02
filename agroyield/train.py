"""Train, validate and test the yield models.

Protocol (no test-set peeking):
  1. Fit each model on the training years, with hyperparameters / boosting rounds chosen on the
     validation years (2021-2022).
  2. Learn blend weights on validation predictions.
  3. Refit every model on train+validation with the chosen settings.
  4. Evaluate once on the held-out test years (2023-2025) and compare with honest baselines.
Targets are log(yield): series differ in scale by up to 1000x and errors are proportional.
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import dataset, metrics
from .config import MODELS
from .features import CATEGORICAL, NUMERIC
from .model import Blend, YieldModel

warnings.filterwarnings("ignore")
SEED = 42


def prep(kind: str):
    """Preprocessing, fitted inside each saved model so the API needs nothing else."""
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    if kind == "native":  # gradient boosting handles NaN itself
        return ColumnTransformer([("c", ordinal, CATEGORICAL), ("n", "passthrough", NUMERIC)])
    if kind == "filled":  # sklearn forests / trees need a value
        return ColumnTransformer([("c", ordinal, CATEGORICAL),
                                  ("n", SimpleImputer(strategy="constant", fill_value=-999.0), NUMERIC)])
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)
    return ColumnTransformer([("c", onehot, CATEGORICAL),
                              ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                  StandardScaler()), NUMERIC)])


def specs(n_jobs: int):
    """name -> (prep kind, builder(rounds), boosting?, training-row cap). Boosting rounds are tuned on validation."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "Linear Regression": ("linear", lambda n: LinearRegression(), False, None),
        "Ridge Regression": ("linear", lambda n: Ridge(alpha=30.0), False, None),
        "Decision Tree": ("filled", lambda n: DecisionTreeRegressor(max_depth=18, min_samples_leaf=20, random_state=SEED), False, None),
        "K-NN": ("linear", lambda n: KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs), False, 60_000),
        "SVR": ("linear", lambda n: SVR(C=3.0, epsilon=0.05, cache_size=1500), False, 25_000),
        "Random Forest": ("filled", lambda n: RandomForestRegressor(
            n_estimators=120, max_depth=22, min_samples_leaf=8, max_samples=0.15, max_features=0.5,
            n_jobs=n_jobs, random_state=SEED), False, None),
        "Hist Gradient Boosting": ("native", lambda n: HistGradientBoostingRegressor(
            max_iter=n, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=SEED), True, None),
        "XGBoost": ("native", lambda n: XGBRegressor(
            n_estimators=n, learning_rate=0.06, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, tree_method="hist", n_jobs=n_jobs, random_state=SEED), True, None),
        "LightGBM": ("native", lambda n: LGBMRegressor(
            n_estimators=n, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=40, n_jobs=n_jobs, random_state=SEED, verbose=-1), True, None),
        "CatBoost": ("native", lambda n: CatBoostRegressor(
            iterations=n, learning_rate=0.1, depth=8, thread_count=n_jobs, random_seed=SEED, verbose=0), True, None),
    }


def _fit(kind, make, rounds, X, y):
    p = prep(kind).fit(X)
    est = make(rounds).fit(p.transform(X), y)
    return p, est


def _staged_val_mae(name, p, est, Xv, yv, rounds):
    """Validation error after each of a few round counts; picks the best without refitting."""
    Xt = p.transform(Xv)
    best, best_mae = rounds, np.inf
    for r in sorted({max(50, int(rounds * f)) for f in (0.25, 0.5, 0.75, 1.0)}):
        if name == "XGBoost":
            pred = est.predict(Xt, iteration_range=(0, r))
        elif name == "LightGBM":
            pred = est.predict(Xt, num_iteration=r)
        elif name == "CatBoost":
            pred = est.predict(Xt, ntree_end=r)
        else:  # Hist GB: staged predict
            pred = list(est.staged_predict(Xt))[r - 1]
        mae = float(np.mean(np.abs(pred - yv)))
        if mae < best_mae:
            best, best_mae = r, mae
    return best


ROUNDS = {"Hist Gradient Boosting": 500, "XGBoost": 800, "LightGBM": 1000, "CatBoost": 1200}


def baselines(train, test):
    """Honest baselines: predictions per test row using only earlier information."""
    out = {"Persistence (last observed yield)": test["prev_yield"].astype("float64"),
           "County history mean": test["hist_mean"].astype("float64")}
    slope, icpt = {}, {}
    for s_, g in train.groupby("series", observed=True):
        if g["year"].nunique() > 5:
            slope[str(s_)], icpt[str(s_)] = np.polyfit(g["year"].astype("float64"), g["yield"].astype("float64"), 1)
    key = test["series"].astype(str)
    out["Series linear trend"] = key.map(slope) * test["year"].astype("float64") + key.map(icpt)
    # Fall back to the series median where a baseline has nothing (e.g. a county's first year)
    med = train.groupby("series", observed=True)["yield"].median()
    fallback = test["series"].map(med).astype("float64")
    return {k: v.fillna(fallback).clip(lower=1e-6) for k, v in out.items()}


def run(variant: str = "A", n_jobs: int = -1, only=None, out_dir=None):
    out_dir = out_dir or MODELS / f"variant_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = dataset.load(variant)
    train, val, test = dataset.split(df)
    trainval = pd.concat([train, val])
    print(f"variant {variant}: train {len(train):,}  val {len(val):,}  test {len(test):,}")
    Xtr, ytr = dataset.xy(train)
    Xva, yva = dataset.xy(val)
    Xtv, ytv = dataset.xy(trainval)
    Xte = test[dataset.FEATURES]
    rng = np.random.default_rng(SEED)

    fitted, val_pred, timings = {}, {}, {}
    for name, (kind, make, boosting, cap) in specs(n_jobs if n_jobs > 0 else 8).items():
        if only and name not in only:
            continue
        t0 = time.time()
        idx = rng.choice(len(Xtr), cap, replace=False) if cap and len(Xtr) > cap else slice(None)
        rounds = ROUNDS.get(name, 0)
        p, est = _fit(kind, make, rounds, Xtr.iloc[idx], ytr.iloc[idx])
        if boosting:
            rounds = _staged_val_mae(name, p, est, Xva, yva, rounds)
        val_pred[name] = np.asarray(est.predict(p.transform(Xva)), "float64")
        # Refit on train+validation (same settings) for the model that is evaluated and shipped
        idx2 = rng.choice(len(Xtv), cap, replace=False) if cap and len(Xtv) > cap else slice(None)
        p2, est2 = _fit(kind, make, rounds, Xtv.iloc[idx2], ytv.iloc[idx2])
        fitted[name] = YieldModel(p2, est2, name, {"rounds": rounds, "train_rows": len(Xtv.iloc[idx2])})
        timings[name] = round(time.time() - t0, 1)
        vm = metrics.summarize(np.exp(yva), np.exp(val_pred[name]))
        print(f"  {name:24s} val MAPE {vm['MAPE']:6.2f}%  rounds {rounds or '-':>5}  {timings[name]}s", flush=True)

    # Blend: non-negative weights on validation predictions of the models trained on train only
    boosters = [n for n in ("Hist Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Random Forest") if n in fitted]
    if len(boosters) >= 2:
        A = np.column_stack([val_pred[n] for n in boosters])
        w, _ = nnls(A, np.asarray(yva))
        w = w / w.sum()
        fitted["Blend"] = Blend([fitted[n] for n in boosters], w)
        print("  blend weights:", {n: round(float(x), 3) for n, x in zip(boosters, w)})

    # Single evaluation on the held-out test years
    base = baselines(trainval, test)
    rows, preds = [], {}
    for name, m in fitted.items():
        preds[name] = m.predict(Xte)
    for name, p in {**base, **preds}.items():
        rows.append({"Model": name, "Type": "Baseline" if name in base else "Model", **metrics.summarize(test["yield"], p),
                     "train_seconds": timings.get(name)})
    table = pd.DataFrame(rows).sort_values("MAPE")
    best = table[table["Type"] == "Model"].iloc[0]["Model"]
    ref = table[table["Type"] == "Baseline"].iloc[0]
    table["skill_vs_best_baseline"] = [metrics.skill(m, ref["MAPE"]) for m in table["MAPE"]]
    table.to_csv(out_dir / "model_comparison.csv", index=False)

    per_commodity = metrics.by_group(test, preds[best], "commodity")
    base_best = base[ref["Model"]]
    per_commodity["baseline_MAPE"] = [
        metrics.summarize(test[test["commodity"] == c]["yield"], base_best[test["commodity"] == c])["MAPE"]
        for c in per_commodity["commodity"]]
    per_commodity["skill"] = [metrics.skill(a, b) for a, b in zip(per_commodity["MAPE"], per_commodity["baseline_MAPE"])]
    per_commodity.to_csv(out_dir / "per_commodity.csv", index=False)
    metrics.by_group(test, preds[best], "series").to_csv(out_dir / "per_series.csv", index=False)

    for name, m in fitted.items():
        joblib.dump(m, out_dir / (name.lower().replace(" ", "_").replace("-", "_") + ".joblib"), compress=3)
    (out_dir / "summary.json").write_text(json.dumps({
        "variant": variant, "best_model": best, "best_baseline": ref["Model"],
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "test_best": table.iloc[0].to_dict()}, indent=2, default=float))
    print(table[["Model", "Type", "MAPE", "MedAPE", "within10", "R2_log", "skill_vs_best_baseline"]].round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "A")
