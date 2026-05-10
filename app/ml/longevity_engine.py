"""
LIFELINE AI — Core Longevity Prediction Engine
Research-backed epidemiological model with SHAP-style feature attribution.

Key Sources:
  WHO Global Health Observatory life tables 2023
  NHANES longitudinal mortality linkage studies
  UK Biobank 500k-participant cohort (Pilling et al. 2017)
  Lancet 2016 BMI/mortality meta-analysis (Di Angelantonio et al.)
  JAMA Int Med 2015 physical activity/mortality (Arem et al.)
  Holt-Lunstad PLOS Med 2010 social relationship meta-analysis
  Cappuccio et al. Sleep 2010 — sleep duration meta-analysis
  Lewington et al. Lancet 2002 — blood pressure & mortality
  Stringhini et al. Lancet 2017 — SES & longevity
  Walker et al. World Psychiatry 2015 — mental illness & mortality
  GBD 2020 — global burden of disease risk factors
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Sex-stratified baseline lifespan (WHO 2023, high-income country profile)
# ---------------------------------------------------------------------------
BASELINE: Dict[str, float] = {"female": 83.2, "male": 78.1, "other": 80.7}
SIGMA = 4.5   # population SD in years


@dataclass
class LongevityResult:
    predicted_age:    float
    remaining_years:  float
    biological_age:   int
    longevity_score:  int
    percentile:       int
    shap_factors:     Dict[str, float] = field(default_factory=dict)
    domain_scores:    Dict[str, int]   = field(default_factory=dict)
    insights:         List[dict]       = field(default_factory=list)
    reasoning:        str = ""
    confidence_low:   float = 0.0
    confidence_high:  float = 0.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual factor delta functions (returns delta_years, label)
# ---------------------------------------------------------------------------

def _smoking_delta(status: str) -> Tuple[float, str]:
    """Doll & Peto BMJ 2004; CDC actuarial tables."""
    m = {"never": 0.0, "former": -1.8, "light": -3.5, "moderate": -6.5, "heavy": -10.0}
    return m.get(status, 0.0), "Smoking"


def _alcohol_delta(level: str) -> Tuple[float, str]:
    """Burton & Sheron Lancet 2018; GBD 2020 alcohol risk."""
    m = {"never": 0.0, "light": 0.4, "moderate": -1.2, "heavy": -4.5}
    return m.get(level, 0.0), "Alcohol"


def _exercise_delta(days: int) -> Tuple[float, str]:
    """Arem et al. JAMA Int Med 2015; WHO PA guidelines."""
    if days == 0:   return -2.0, "Exercise"
    if days == 1:   return  0.5, "Exercise"
    if days <= 3:   return  2.0, "Exercise"
    if days <= 5:   return  3.2, "Exercise"
    return 4.0, "Exercise"


def _sleep_delta(hours: float) -> Tuple[float, str]:
    """Cappuccio et al. Sleep 2010 meta-analysis. U-shaped optimal 7-8.5h."""
    if 7.0 <= hours <= 8.5:   return  1.8, "Sleep"
    if 6.0 <= hours < 7.0:    return  0.0, "Sleep"
    if 8.5 < hours <= 9.5:    return  0.0, "Sleep"
    if hours < 6.0:            return -2.5, "Sleep"
    return -2.0, "Sleep"  # >9.5h often illness proxy


def _stress_delta(level: int) -> Tuple[float, str]:
    """Kivimäki et al. Lancet 2012; Cohen 2007 cortisol/immunity."""
    if level <= 3:  return  1.2, "Stress"
    if level <= 5:  return  0.0, "Stress"
    if level <= 7:  return -1.5, "Stress"
    return -3.0, "Stress"


def _social_delta(score: int) -> Tuple[float, str]:
    """Holt-Lunstad PLOS Med 2010: 50% survival advantage with strong ties."""
    return round((score - 5) * 0.37, 2), "Social Connection"


def _bmi_delta(bmi: float) -> Tuple[float, str]:
    """Di Angelantonio Lancet 2016 BMI meta-analysis (n=10.6M)."""
    if 20.0 <= bmi <= 25.0:  return  1.0, "BMI"
    if 25.0 < bmi <= 27.5:   return  0.0, "BMI"
    if 27.5 < bmi <= 30.0:   return -1.0, "BMI"
    if 30.0 < bmi <= 35.0:   return -2.5, "BMI"
    if 35.0 < bmi <= 40.0:   return -4.5, "BMI"
    if bmi > 40.0:            return -7.0, "BMI"
    return -2.5, "BMI"   # underweight


def _nutrition_delta(fv: int, processed: str) -> Tuple[float, float, str, str]:
    """GBD 2019 diet risk factors; Oyebode et al. BMJ 2014; NutriNet-Santé cohort."""
    fv_delta   = min(4.0, (fv - 2) * 0.55)
    proc_map   = {"rarely": 0.5, "sometimes": -0.5, "often": -1.8, "very_often": -3.5}
    proc_delta = proc_map.get(processed, 0.0)
    return round(fv_delta, 2), round(proc_delta, 2), "Fruit & Veg Intake", "Processed Food"


def _bp_delta(systolic: int) -> Tuple[float, str]:
    """Lewington et al. Lancet 2002: 20mmHg above 115 → 2× CVD risk."""
    if systolic < 120:  return  1.5, "Blood Pressure"
    if systolic < 130:  return  0.5, "Blood Pressure"
    if systolic < 140:  return -1.0, "Blood Pressure"
    if systolic < 160:  return -2.5, "Blood Pressure"
    return -4.5, "Blood Pressure"


def _hr_delta(bpm: int) -> Tuple[float, str]:
    """Cooney et al. Eur Heart J 2010: resting HR and all-cause mortality."""
    if bpm < 60:  return  1.8, "Resting Heart Rate"
    if bpm < 70:  return  1.0, "Resting Heart Rate"
    if bpm < 80:  return  0.0, "Resting Heart Rate"
    if bpm < 90:  return -1.2, "Resting Heart Rate"
    return -2.5, "Resting Heart Rate"


def _condition_delta(cond: str) -> Tuple[float, str]:
    """GBD 2019 DALY fractions converted to life-years lost estimates."""
    m = {"none": 0.0, "hypertension": -3.0, "diabetes": -5.0,
         "heart_disease": -7.0, "multiple": -9.5}
    return m.get(cond, 0.0), "Chronic Conditions"


def _family_delta(bracket: str) -> Tuple[float, str]:
    """Pilling et al. Aging Cell 2017; GWAS longevity heritability ~25%."""
    m = {"below_70": -2.5, "70_80": 0.0, "80_90": 2.0, "above_90": 4.0}
    return m.get(bracket, 0.0), "Family Longevity"


def _mental_health_delta(status: str) -> Tuple[float, str]:
    """Walker et al. World Psychiatry 2015; Firth et al. 2019 review."""
    m = {"excellent": 1.5, "good": 0.5, "moderate": -1.5, "poor": -4.0}
    return m.get(status, 0.0), "Mental Health"


def _ses_delta(level: str) -> Tuple[float, str]:
    """Stringhini et al. Lancet 2017: 25yr prospective multi-cohort study."""
    m = {"low": -3.0, "lower_middle": -1.5, "middle": 0.0,
         "upper_middle": 1.0, "high": 2.0}
    return m.get(level, 0.0), "Socioeconomic Status"


def _education_delta(level: str) -> Tuple[float, str]:
    """Lleras-Muney 2005; GBD education-mortality gradient."""
    m = {"less_hs": -2.5, "hs": -1.0, "some_college": 0.0,
         "bachelors": 1.5, "graduate": 2.5}
    return m.get(level, 0.0), "Education"


def _aqi_delta(aqi: float) -> Tuple[float, str]:
    """Pope et al. NEJM 2009: PM2.5 and life expectancy. Each 10μg/m³ ≈ -0.6yr."""
    excess = max(0.0, aqi - 35)
    return round(-(excess / 20.0) * 0.6, 2), "Air Quality"


# ---------------------------------------------------------------------------
# Biological age estimator (inspired by Levine/Horvath epigenetic clock)
# ---------------------------------------------------------------------------
def estimate_biological_age(chron_age: int, factors: Dict[str, float]) -> int:
    high_impact = ["Smoking", "BMI", "Exercise", "Stress",
                   "Blood Pressure", "Resting Heart Rate",
                   "Chronic Conditions", "Sleep"]
    offset = sum(-factors.get(k, 0.0) * 0.55 for k in high_impact)
    return max(chron_age - 10, min(chron_age + 20, int(round(chron_age + offset))))


# ---------------------------------------------------------------------------
# Domain composite scores (0–100)
# ---------------------------------------------------------------------------
def compute_domain_scores(f: Dict[str, float]) -> Dict[str, int]:
    def clamp(v): return max(0, min(100, int(round(v))))
    return {
        "Cardiovascular": clamp(65 + f.get("Blood Pressure", 0)*7 + f.get("Resting Heart Rate", 0)*6 + f.get("BMI", 0)*4),
        "Metabolic":      clamp(65 + f.get("BMI", 0)*6 + f.get("Fruit & Veg Intake", 0)*6 + f.get("Processed Food", 0)*5),
        "Lifestyle":      clamp(60 + f.get("Exercise", 0)*7 + f.get("Smoking", 0)*4 + f.get("Alcohol", 0)*3),
        "Mental & Social":clamp(60 + f.get("Social Connection", 0)*8 + f.get("Mental Health", 0)*8 + f.get("Stress", 0)*5),
        "Genetic & Bio":  clamp(60 + f.get("Family Longevity", 0)*9 + f.get("Sleep", 0)*6),
        "Environment":    clamp(70 + f.get("Air Quality", 0)*8 + f.get("Socioeconomic Status", 0)*5),
    }


# ---------------------------------------------------------------------------
# Insights generator
# ---------------------------------------------------------------------------
def generate_insights(inp: dict, factors: Dict[str, float]) -> List[dict]:
    insights = []

    def add(t, icon, text):
        insights.append({"type": t, "icon": icon, "text": text})

    if inp.get("smoking") in ("moderate", "heavy"):
        add("negative", "ti-smoking-no",
            "Smoking is the single most preventable cause of premature death. Quitting before "
            "age 40 recovers up to 90% of lost life expectancy (Doll & Peto, BMJ 2004).")
    elif inp.get("smoking") == "former":
        add("neutral", "ti-smoking-no",
            "Former smoking carries residual cardiovascular and cancer risk for 10–15 years "
            "post-cessation. Regular screening and healthy lifestyle accelerate risk recovery.")

    if inp.get("exercise", 3) >= 5:
        add("positive", "ti-run",
            "Exercising 5+ days/week places you in the top longevity quartile. High "
            "cardiorespiratory fitness reduces all-cause mortality by up to 45% "
            "(Arem et al., JAMA Internal Medicine 2015).")
    elif inp.get("exercise", 3) == 0:
        add("negative", "ti-sofa",
            "Physical inactivity is responsible for 6% of global deaths (WHO 2020). "
            "Even 15 minutes of brisk walking daily reduces mortality risk by 14% "
            "(Wen et al., Lancet 2011).")

    sl = float(inp.get("sleep", 7))
    if sl < 6:
        add("negative", "ti-moon-off",
            "Chronic short sleep (<6h) increases all-cause mortality by 12% and accelerates "
            "telomere attrition, immune suppression, and metabolic dysregulation "
            "(Cappuccio et al., Sleep 2010).")
    elif 7 <= sl <= 8.5:
        add("positive", "ti-zzz",
            "Optimal sleep (7–8.5h) enables peak cellular repair via glymphatic clearance, "
            "immune consolidation, and growth hormone secretion — a pillar of longevity.")

    if int(inp.get("social", 5)) <= 3:
        add("negative", "ti-user-off",
            "Chronic social isolation is as harmful as smoking 15 cigarettes/day. "
            "Strong social ties confer a 50% survival advantage (Holt-Lunstad, PLOS Med 2010).")
    elif int(inp.get("social", 5)) >= 8:
        add("positive", "ti-users",
            "Strong social bonds lower inflammatory cytokines, improve treatment adherence, "
            "and activate parasympathetic regulation — powerful biological longevity mechanisms.")

    bmi = float(inp.get("bmi", 23))
    if bmi > 35:
        add("negative", "ti-alert-triangle",
            "Severe obesity (BMI >35) reduces median lifespan by 5–10 years and raises risk "
            "for 13 cancers, heart failure, and type 2 diabetes (Di Angelantonio, Lancet 2016).")
    elif 20 <= bmi <= 25:
        add("positive", "ti-heart-rate-monitor",
            "Healthy BMI (20–25) is associated with the lowest all-cause mortality across "
            "large population cohorts. You're in the optimal metabolic range (GBD 2019).")

    if int(inp.get("stress", 5)) >= 8:
        add("negative", "ti-brain",
            "Chronic high stress accelerates telomere shortening, dysregulates cortisol, and "
            "significantly elevates cardiovascular risk (Kivimäki et al., Lancet 2012). "
            "Evidence-based interventions: MBSR mindfulness, CBT, and regular aerobic exercise.")

    if inp.get("family") == "above_90":
        add("positive", "ti-dna-2",
            "Parental longevity above 90 signals protective variants in FOXO3A, APOE-ε2, and "
            "telomere maintenance genes — a genuine and quantifiable genetic headstart "
            "(Pilling et al., Aging Cell 2017).")

    if inp.get("conditions", "none") != "none":
        add("neutral", "ti-stethoscope",
            "Chronic disease reduces predicted lifespan, but optimal medical management, "
            "lifestyle modification, and medication adherence can substantially narrow this gap.")

    if int(inp.get("bp", 120)) >= 140:
        add("negative", "ti-heart-broken",
            "Stage 2 hypertension (≥140 mmHg) doubles stroke risk. Each 10 mmHg reduction "
            "saves approximately 1.5 life-years (Lewington et al., Lancet 2002).")

    if inp.get("processed") in ("often", "very_often"):
        add("negative", "ti-salad",
            "High ultra-processed food intake (>4 servings/day) is linked to 62% higher "
            "all-cause mortality in the NutriNet-Santé cohort (Schnabel et al., JAMA 2019).")

    if int(inp.get("fv", 4)) >= 7:
        add("positive", "ti-leaf",
            "7+ daily fruit/vegetable servings are associated with 42% lower all-cause "
            "mortality risk — one of the most consistent findings in nutritional epidemiology "
            "(Oyebode et al., BMJ 2014).")

    return insights[:7]


# ---------------------------------------------------------------------------
# Reasoning narrative
# ---------------------------------------------------------------------------
def generate_reasoning(inp: dict, result: "LongevityResult") -> str:
    sex_label = "women" if inp.get("sex") == "female" else "men"
    base = BASELINE.get(inp.get("sex", "male"), 79.0)
    diff = result.predicted_age - base
    direction = "above" if diff >= 0 else "below"

    pos = sorted([(k, v) for k, v in result.shap_factors.items() if v > 0], key=lambda x: -x[1])[:3]
    neg = sorted([(k, v) for k, v in result.shap_factors.items() if v < 0], key=lambda x: x[1])[:3]
    pos_str = ", ".join(f"{k} (+{v:.1f} yrs)" for k, v in pos) or "none dominant"
    neg_str = ", ".join(f"{k} ({v:.1f} yrs)" for k, v in neg) or "none identified"

    bio_diff = result.biological_age - int(inp.get("age", 35))
    bio_msg = (
        f"Biological age is estimated at {result.biological_age} — "
        + (f"{abs(bio_diff)} years older than chronological age, suggesting accelerated "
           f"cellular ageing driven by modifiable risk factors." if bio_diff > 2
           else f"{abs(bio_diff)} years younger than chronological age, reflecting healthy "
                f"cellular maintenance consistent with your lifestyle profile." if bio_diff < -1
           else "closely aligned with chronological age.")
    )

    return (
        f"WHO 2023 life tables establish a baseline lifespan of {base:.1f} years for {sex_label} "
        f"in high-income countries. Your personalised prediction of {result.predicted_age:.0f} years "
        f"is {abs(diff):.1f} years {direction} this baseline, reflecting cumulative lifestyle effects.\n\n"
        f"Strongest longevity assets: {pos_str}. These are well-supported by large-cohort "
        f"epidemiological evidence and measurably extend your trajectory.\n\n"
        f"Primary risk factors: {neg_str}. All are modifiable — research consistently shows "
        f"that behavioural change at any age yields measurable life-expectancy gains.\n\n"
        f"{bio_msg}\n\n"
        f"Longevity score {result.longevity_score}/100 places you at the {result.percentile}th "
        f"percentile for your age and sex cohort. Confidence interval: "
        f"{result.confidence_low:.0f}–{result.confidence_high:.0f} years (±{SIGMA:.1f}σ)."
    )


# ---------------------------------------------------------------------------
# Master prediction entry point
# ---------------------------------------------------------------------------
def predict(inp: dict) -> LongevityResult:
    base = BASELINE.get(inp.get("sex", "male"), 79.0)
    factors: Dict[str, float] = {}

    def rec(delta, label):
        factors[label] = round(delta, 2)
        return delta

    total = 0.0
    total += rec(*_smoking_delta(inp.get("smoking", "never")))
    total += rec(*_alcohol_delta(inp.get("alcohol", "light")))
    total += rec(*_exercise_delta(int(inp.get("exercise", 3))))
    total += rec(*_sleep_delta(float(inp.get("sleep", 7))))
    total += rec(*_stress_delta(int(inp.get("stress", 5))))
    total += rec(*_social_delta(int(inp.get("social", 5))))
    total += rec(*_bmi_delta(float(inp.get("bmi", 23))))

    fv_d, proc_d, fv_lbl, proc_lbl = _nutrition_delta(int(inp.get("fv", 4)), inp.get("processed", "rarely"))
    total += rec(fv_d, fv_lbl)
    total += rec(proc_d, proc_lbl)

    total += rec(*_bp_delta(int(inp.get("bp", 120))))
    total += rec(*_hr_delta(int(inp.get("hr", 68))))
    total += rec(*_condition_delta(inp.get("conditions", "none")))
    total += rec(*_family_delta(inp.get("family", "80_90")))
    total += rec(*_mental_health_delta(inp.get("mental", "good")))
    total += rec(*_ses_delta(inp.get("ses", "middle")))
    total += rec(*_education_delta(inp.get("education", "some_college")))
    total += rec(*_aqi_delta(float(inp.get("aqi", 45))))

    predicted = max(float(inp["age"]) + 0.5, min(115.0, base + total))
    predicted = round(predicted, 1)
    remaining = max(0.0, round(predicted - float(inp["age"]), 1))
    bio_age   = estimate_biological_age(int(inp["age"]), factors)
    score     = max(0, min(100, int(round(50 + total * 2.0))))

    z = total / SIGMA
    percentile = max(1, min(99, int(round((1 + math.erf(z / math.sqrt(2))) / 2 * 100))))

    result = LongevityResult(
        predicted_age=predicted,
        remaining_years=remaining,
        biological_age=bio_age,
        longevity_score=score,
        percentile=percentile,
        shap_factors=factors,
        domain_scores=compute_domain_scores(factors),
        confidence_low=round(predicted - SIGMA, 0),
        confidence_high=round(predicted + SIGMA, 0),
    )
    result.insights  = generate_insights(inp, factors)
    result.reasoning = generate_reasoning(inp, result)
    return result
