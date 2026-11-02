import numpy as np

from agroyield import metrics


def test_perfect_predictions_score_perfectly():
    y = np.array([10.0, 20.0, 40.0])
    m = metrics.summarize(y, y)
    assert m["MAPE"] == 0 and m["within10"] == 100 and m["logRMSE"] == 0 and m["R2_log"] == 1


def test_metrics_are_scale_free():
    y, p = np.array([10.0, 20.0, 40.0]), np.array([11.0, 18.0, 44.0])
    a, b = metrics.summarize(y, p), metrics.summarize(y * 1000, p * 1000)
    assert abs(a["MAPE"] - b["MAPE"]) < 1e-9 and abs(a["logRMSE"] - b["logRMSE"]) < 1e-9


def test_skill_is_relative_to_the_baseline():
    assert metrics.skill(5, 10) == 0.5 and metrics.skill(10, 10) == 0 and metrics.skill(20, 10) < 0
