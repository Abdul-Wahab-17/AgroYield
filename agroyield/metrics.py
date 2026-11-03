"""Scale-free evaluation. Series differ in units by up to 1000x (bu/acre vs lbs/acre), so pooled
RMSE/MAE are dominated by whichever crop has the biggest numbers. These metrics compare like with like."""
import numpy as np
import pandas as pd


def summarize(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, "float64"), np.asarray(y_pred, "float64")
    ape = np.abs(y_pred - y_true) / y_true
    lt, lp = np.log(y_true), np.log(np.clip(y_pred, 1e-9, None))
    ss_res, ss_tot = ((lt - lp) ** 2).sum(), ((lt - lt.mean()) ** 2).sum()
    return {"n": int(len(y_true)), "MAPE": float(100 * ape.mean()), "MedAPE": float(100 * np.median(ape)),
            "within10": float(100 * (ape <= 0.10).mean()), "logRMSE": float(np.sqrt(((lt - lp) ** 2).mean())),
            "R2_log": float(1 - ss_res / ss_tot)}


def by_group(df: pd.DataFrame, pred, group: str) -> pd.DataFrame:
    rows = []
    for key, g in df.assign(_pred=pred).groupby(group, observed=True):
        m = summarize(g["yield"], g["_pred"])
        rows.append({group: key, **m})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def skill(model_mape: float, baseline_mape: float) -> float:
    """Fraction of the baseline's error removed (1 = perfect, 0 = no better, <0 = worse)."""
    return float(1 - model_mape / baseline_mape)
