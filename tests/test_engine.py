"""Unit tests for the longevity prediction engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.longevity_engine import predict, BASELINE

BASE_MALE   = BASELINE["male"]
BASE_FEMALE = BASELINE["female"]

HEALTHY_MALE = {
    "age": 35, "sex": "male", "education": "graduate", "ses": "upper_middle",
    "smoking": "never", "alcohol": "light", "exercise": 5, "sleep": 7.5,
    "stress": 3, "social": 9, "fv": 8, "processed": "rarely",
    "bmi": 22, "hr": 58, "bp": 115, "conditions": "none",
    "family": "above_90", "mental": "excellent", "env": "suburban", "aqi": 30,
}

UNHEALTHY_MALE = {
    "age": 45, "sex": "male", "education": "less_hs", "ses": "low",
    "smoking": "heavy", "alcohol": "heavy", "exercise": 0, "sleep": 5,
    "stress": 9, "social": 2, "fv": 1, "processed": "very_often",
    "bmi": 38, "hr": 92, "bp": 155, "conditions": "multiple",
    "family": "below_70", "mental": "poor", "env": "high_pollution", "aqi": 220,
}


def test_healthy_above_baseline():
    r = predict(HEALTHY_MALE)
    assert r.predicted_age > BASE_MALE, f"Expected > {BASE_MALE}, got {r.predicted_age}"

def test_unhealthy_below_baseline():
    r = predict(UNHEALTHY_MALE)
    assert r.predicted_age < BASE_MALE, f"Expected < {BASE_MALE}, got {r.predicted_age}"

def test_healthy_outperforms_unhealthy():
    rh = predict(HEALTHY_MALE)
    ru = predict(UNHEALTHY_MALE)
    assert rh.predicted_age > ru.predicted_age

def test_predicted_age_above_input_age():
    r = predict(HEALTHY_MALE)
    assert r.predicted_age > HEALTHY_MALE["age"]

def test_remaining_years_consistent():
    r = predict(HEALTHY_MALE)
    assert abs(r.predicted_age - HEALTHY_MALE["age"] - r.remaining_years) < 1

def test_score_range():
    for inp in (HEALTHY_MALE, UNHEALTHY_MALE):
        r = predict(inp)
        assert 0 <= r.longevity_score <= 100

def test_percentile_range():
    for inp in (HEALTHY_MALE, UNHEALTHY_MALE):
        r = predict(inp)
        assert 1 <= r.percentile <= 99

def test_female_baseline_higher():
    male_inp   = {**HEALTHY_MALE, "sex": "male"}
    female_inp = {**HEALTHY_MALE, "sex": "female"}
    rm = predict(male_inp)
    rf = predict(female_inp)
    assert rf.predicted_age > rm.predicted_age

def test_insights_non_empty():
    r = predict(UNHEALTHY_MALE)
    assert len(r.insights) > 0

def test_reasoning_non_empty():
    r = predict(HEALTHY_MALE)
    assert len(r.reasoning) > 100

def test_shap_keys_present():
    r = predict(HEALTHY_MALE)
    assert "Exercise" in r.shap_factors
    assert "Smoking"  in r.shap_factors

def test_domain_scores_all_six():
    r = predict(HEALTHY_MALE)
    assert len(r.domain_scores) == 6
    for v in r.domain_scores.values():
        assert 0 <= v <= 100

def test_confidence_interval_ordered():
    r = predict(HEALTHY_MALE)
    assert r.confidence_low < r.predicted_age < r.confidence_high


if __name__ == "__main__":
    tests = [v for k,v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
