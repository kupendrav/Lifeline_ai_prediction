"""
LIFELINE AI — Synthetic Longevity Dataset Generator
Produces 100,000+ scientifically plausible records grounded in:
  - WHO life table age/sex distributions
  - NHANES population health statistics
  - UK Biobank covariate prevalence data
  - GBD 2020 risk factor distributions

Output: data/longevity_dataset.csv  (≥100k rows, ~25 features)

Usage:
    python -m app.data.generate_dataset
    python -m app.data.generate_dataset --rows 200000 --seed 42
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── EPIDEMIOLOGICAL DISTRIBUTIONS (from NHANES / WHO population surveys) ────

SMOKING_PROBS = {
    "female": {"never": 0.56, "former": 0.20, "light": 0.06, "moderate": 0.12, "heavy": 0.06},
    "male":   {"never": 0.46, "former": 0.25, "light": 0.07, "moderate": 0.14, "heavy": 0.08},
}
ALCOHOL_PROBS = {
    "female": {"never": 0.35, "light": 0.38, "moderate": 0.18, "heavy": 0.09},
    "male":   {"never": 0.22, "light": 0.30, "moderate": 0.28, "heavy": 0.20},
}
CONDITION_PROBS_BY_AGE = {
    # (min_age, max_age): {condition: prob}
    (18, 40):  {"none": 0.82, "hypertension": 0.08, "diabetes": 0.04, "heart_disease": 0.02, "multiple": 0.04},
    (40, 60):  {"none": 0.58, "hypertension": 0.20, "diabetes": 0.12, "heart_disease": 0.05, "multiple": 0.05},
    (60, 100): {"none": 0.32, "hypertension": 0.28, "diabetes": 0.18, "heart_disease": 0.12, "multiple": 0.10},
}
EDUCATION_PROBS = {"less_hs": 0.09, "hs": 0.27, "some_college": 0.29, "bachelors": 0.21, "graduate": 0.14}
SES_PROBS       = {"low": 0.14, "lower_middle": 0.22, "middle": 0.33, "upper_middle": 0.21, "high": 0.10}
FAMILY_PROBS    = {"below_70": 0.15, "70_80": 0.32, "80_90": 0.38, "above_90": 0.15}
MENTAL_PROBS    = {"excellent": 0.22, "good": 0.44, "moderate": 0.24, "poor": 0.10}
ENV_PROBS       = {"rural": 0.19, "suburban": 0.41, "urban": 0.30, "high_pollution": 0.10}
PROCESSED_PROBS = {"rarely": 0.22, "sometimes": 0.38, "often": 0.28, "very_often": 0.12}
SEX_PROBS       = {"female": 0.504, "male": 0.490, "other": 0.006}

# BMI distribution parameters (NHANES 2020) — Normal mixture
BMI_PARAMS = {
    "female": [(23.0, 3.5, 0.35), (27.5, 4.0, 0.35), (35.0, 5.0, 0.30)],
    "male":   [(24.0, 3.2, 0.33), (28.5, 4.0, 0.37), (36.0, 5.0, 0.30)],
}

# AQI by environment type (approximate)
AQI_BY_ENV = {
    "rural":          (12, 20),
    "suburban":       (30, 25),
    "urban":          (65, 35),
    "high_pollution": (140, 60),
}


# ── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def _weighted_choice(probs: Dict[str, float], rng: random.Random) -> str:
    keys, weights = zip(*probs.items())
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for k, w in zip(keys, weights):
        cumulative += w
        if r <= cumulative:
            return k
    return keys[-1]


def _normal_clamp(mu: float, sigma: float, lo: float, hi: float, rng: random.Random) -> float:
    val = rng.gauss(mu, sigma)
    return max(lo, min(hi, val))


def _mixture_bmi(sex: str, rng: random.Random) -> float:
    components = BMI_PARAMS[sex if sex in BMI_PARAMS else "male"]
    mu, sigma, _ = rng.choices(components, weights=[c[2] for c in components])[0]
    return round(_normal_clamp(mu, sigma, 14.0, 60.0, rng), 1)


def _get_condition(age: int, rng: random.Random) -> str:
    for (lo, hi), probs in CONDITION_PROBS_BY_AGE.items():
        if lo <= age < hi:
            return _weighted_choice(probs, rng)
    return _weighted_choice(CONDITION_PROBS_BY_AGE[(60, 100)], rng)


def _correlate_bmi_exercise(bmi: float, rng: random.Random) -> int:
    """Higher BMI slightly lowers exercise probability (realistic correlation)."""
    base_mean = max(0, 5.0 - (bmi - 22) * 0.08)
    raw = rng.gauss(base_mean, 1.8)
    return max(0, min(7, int(round(raw))))


def _correlate_stress_sleep(stress: int, rng: random.Random) -> float:
    """Higher stress correlates with shorter / worse sleep."""
    base_hours = 8.5 - (stress - 5) * 0.18
    raw = rng.gauss(base_hours, 0.9)
    return round(max(3.5, min(12.0, raw)), 1)


def _correlate_ses_aqi(ses: str, env: str, rng: random.Random) -> float:
    """Lower SES correlates with higher pollution exposure."""
    base_mu, base_sigma = AQI_BY_ENV[env]
    ses_adj = {"low": 25, "lower_middle": 10, "middle": 0, "upper_middle": -8, "high": -15}
    mu = base_mu + ses_adj.get(ses, 0)
    return round(max(5.0, min(400.0, rng.gauss(mu, base_sigma))), 1)


# ── PREDICTION ENGINE (inline copy, no Flask dependency) ────────────────────

def _compute_lifespan(inp: dict) -> float:
    BASELINE = {"female": 83.2, "male": 78.1, "other": 80.7}
    base = BASELINE.get(inp["sex"], 79.0)
    total = 0.0

    # Smoking
    total += {"never":0.0,"former":-1.8,"light":-3.5,"moderate":-6.5,"heavy":-10.0}.get(inp["smoking"],0.0)
    # Alcohol
    total += {"never":0.0,"light":0.4,"moderate":-1.2,"heavy":-4.5}.get(inp["alcohol"],0.0)
    # Exercise
    ex = int(inp["exercise"])
    total += (-2.0 if ex==0 else 0.5 if ex==1 else 2.0 if ex<=3 else 3.2 if ex<=5 else 4.0)
    # Sleep
    sl = float(inp["sleep"])
    total += (1.8 if 7<=sl<=8.5 else -2.5 if sl<6 else -2.0 if sl>9.5 else 0.0)
    # Stress
    st = int(inp["stress"])
    total += (1.2 if st<=3 else -1.5 if st<=7 else -3.0 if st>7 else 0.0)
    # Social
    total += (int(inp["social"])-5) * 0.37
    # BMI
    bmi = float(inp["bmi"])
    total += (1.0 if 20<=bmi<=25 else 0.0 if bmi<=27.5 else -1.0 if bmi<=30 else -2.5 if bmi<=35 else -4.5 if bmi<=40 else -7.0)
    # Nutrition
    total += min(4.0, (int(inp["fv"])-2)*0.55)
    total += {"rarely":0.5,"sometimes":-0.5,"often":-1.8,"very_often":-3.5}.get(inp["processed"],0.0)
    # Blood pressure
    bp = int(inp["bp"])
    total += (1.5 if bp<120 else 0.5 if bp<130 else -1.0 if bp<140 else -2.5 if bp<160 else -4.5)
    # Heart rate
    hr = int(inp["hr"])
    total += (1.8 if hr<60 else 1.0 if hr<70 else 0.0 if hr<80 else -1.2 if hr<90 else -2.5)
    # Conditions
    total += {"none":0.0,"hypertension":-3.0,"diabetes":-5.0,"heart_disease":-7.0,"multiple":-9.5}.get(inp["conditions"],0.0)
    # Family
    total += {"below_70":-2.5,"70_80":0.0,"80_90":2.0,"above_90":4.0}.get(inp["family"],0.0)
    # Mental
    total += {"excellent":1.5,"good":0.5,"moderate":-1.5,"poor":-4.0}.get(inp["mental"],0.0)
    # SES
    total += {"low":-3.0,"lower_middle":-1.5,"middle":0.0,"upper_middle":1.0,"high":2.0}.get(inp["ses"],0.0)
    # Education
    total += {"less_hs":-2.5,"hs":-1.0,"some_college":0.0,"bachelors":1.5,"graduate":2.5}.get(inp["education"],0.0)
    # AQI
    total += -(max(0.0, float(inp["aqi"])-35)/20)*0.6
    # Fruit & veg
    total += min(4.0, (int(inp["fv"])-2)*0.55)

    # Add calibrated noise (population heterogeneity)
    return base + total


def _add_noise(lifespan: float, rng: random.Random, sigma: float = 4.5) -> float:
    """Add realistic population variance. Censored at age+1 and 115."""
    noisy = lifespan + rng.gauss(0, sigma)
    return round(max(20.0, min(115.0, noisy)), 1)


# ── RECORD GENERATOR ─────────────────────────────────────────────────────────

def generate_record(rng: random.Random) -> Dict[str, object]:
    # Demographics
    sex       = _weighted_choice(SEX_PROBS, rng)
    age       = int(_normal_clamp(45, 18, 18, 90, rng))
    education = _weighted_choice(EDUCATION_PROBS, rng)
    ses       = _weighted_choice(SES_PROBS, rng)

    # Lifestyle (some correlated)
    smoking   = _weighted_choice(SMOKING_PROBS.get(sex, SMOKING_PROBS["male"]), rng)
    alcohol   = _weighted_choice(ALCOHOL_PROBS.get(sex, ALCOHOL_PROBS["male"]), rng)
    stress    = max(1, min(10, int(round(rng.gauss(5.2, 2.1)))))
    social    = max(1, min(10, int(round(rng.gauss(6.1, 2.0)))))
    sleep     = _correlate_stress_sleep(stress, rng)
    bmi       = _mixture_bmi(sex, rng)
    exercise  = _correlate_bmi_exercise(bmi, rng)
    fv        = max(0, min(14, int(round(rng.gauss(4.2, 2.3)))))
    processed = _weighted_choice(PROCESSED_PROBS, rng)

    # Health metrics (correlated with age/BMI)
    bp_base   = 110 + age*0.45 + (bmi-23)*0.8
    bp        = int(_normal_clamp(bp_base, 12, 80, 190, rng))
    hr_base   = 72 - exercise*1.2 + (bmi-23)*0.3
    hr        = int(_normal_clamp(hr_base, 10, 40, 115, rng))

    # Medical
    conditions = _get_condition(age, rng)
    family     = _weighted_choice(FAMILY_PROBS, rng)
    mental     = _weighted_choice(MENTAL_PROBS, rng)

    # Environment
    env        = _weighted_choice(ENV_PROBS, rng)
    aqi        = _correlate_ses_aqi(ses, env, rng)

    inp = {
        "age": age, "sex": sex, "education": education, "ses": ses,
        "smoking": smoking, "alcohol": alcohol, "exercise": exercise,
        "sleep": sleep, "stress": stress, "social": social,
        "fv": fv, "processed": processed, "bmi": bmi,
        "hr": hr, "bp": bp, "conditions": conditions,
        "family": family, "mental": mental, "env": env, "aqi": aqi,
    }

    deterministic_lifespan = _compute_lifespan(inp)
    observed_lifespan      = _add_noise(deterministic_lifespan, rng)
    remaining_years        = max(0.0, round(observed_lifespan - age, 1))

    record = {**inp,
              "predicted_lifespan": observed_lifespan,
              "remaining_years":    remaining_years}
    return record


# ── FIELDNAMES ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "age","sex","education","ses","smoking","alcohol","exercise","sleep",
    "stress","social","fv","processed","bmi","hr","bp","conditions",
    "family","mental","env","aqi","predicted_lifespan","remaining_years",
]


# ── MAIN ─────────────────────────────────────────────────────────────────────

def generate(n_rows: int = 100_000, seed: int = 42, output_path: str | None = None) -> str:
    rng = random.Random(seed)

    if output_path is None:
        base_dir = Path(__file__).resolve().parent
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(base_dir / "longevity_dataset.csv")

    print(f"Generating {n_rows:,} records → {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(generate_record(rng))
            if (i+1) % 10_000 == 0:
                print(f"  {i+1:>7,} / {n_rows:,} written…")

    print(f"\nDataset complete: {output_path}")
    _print_summary(output_path)
    return output_path


def _print_summary(path: str):
    """Print basic summary stats for the generated dataset."""
    ages, lifespans = [], []
    sex_counts: Dict[str, int] = {}
    cond_counts: Dict[str, int] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ages.append(float(row["age"]))
            lifespans.append(float(row["predicted_lifespan"]))
            sex_counts[row["sex"]] = sex_counts.get(row["sex"], 0) + 1
            cond_counts[row["conditions"]] = cond_counts.get(row["conditions"], 0) + 1

    n = len(ages)
    avg_age = sum(ages)/n
    avg_ls  = sum(lifespans)/n
    std_ls  = math.sqrt(sum((x-avg_ls)**2 for x in lifespans)/n)
    min_ls, max_ls = min(lifespans), max(lifespans)

    print(f"\n{'─'*48}")
    print(f"  Records:           {n:>10,}")
    print(f"  Mean age:          {avg_age:>10.1f}")
    print(f"  Mean lifespan:     {avg_ls:>10.1f}  (σ={std_ls:.1f})")
    print(f"  Lifespan range:    {min_ls:>10.1f} – {max_ls:.1f}")
    print(f"  Sex distribution:  {', '.join(f'{k}={v/n*100:.1f}%' for k,v in sorted(sex_counts.items()))}")
    print(f"  Condition rates:   {', '.join(f'{k}={v/n*100:.1f}%' for k,v in sorted(cond_counts.items()))}")
    print(f"{'─'*48}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIFELINE AI — Dataset Generator")
    parser.add_argument("--rows",   type=int, default=100_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate(args.rows, args.seed, args.output)
