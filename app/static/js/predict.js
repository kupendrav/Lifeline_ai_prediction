/**
 * LIFELINE AI — Frontend Prediction Script
 * Handles form submission, loading state, and redirect to results page.
 */

const LOADING_MESSAGES = [
  "Applying WHO epidemiological risk coefficients...",
  "Calibrating against UK Biobank cohort data...",
  "Running SHAP feature attribution...",
  "Estimating biological age offset...",
  "Computing longevity percentile vs peers...",
  "Generating personalised insights...",
];

function syncVal(inputId, badgeId, formatter) {
  const val = document.getElementById(inputId).value;
  document.getElementById(badgeId).textContent = formatter ? formatter(val) : val;
}

function collectForm() {
  const form = document.getElementById("predict-form");
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });
  // Also grab range inputs (FormData misses disabled ones)
  form.querySelectorAll("input[type=range]").forEach(el => { data[el.name] = el.value; });
  return data;
}

async function submitPrediction() {
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;

  const overlay = document.getElementById("ll-loading");
  overlay.style.display = "flex";

  // Cycle loading messages
  const msgEl = document.getElementById("ll-load-msg");
  let i = 0;
  const timer = setInterval(() => {
    msgEl.textContent = LOADING_MESSAGES[i++ % LOADING_MESSAGES.length];
  }, 800);

  try {
    const payload = collectForm();
    const csrfToken = document.querySelector("input[name=\"csrf_token\"]")?.value;
    const headers = { "Content-Type": "application/json" };
    if (csrfToken) headers["X-CSRFToken"] = csrfToken;
    const resp = await fetch("/api/v1/predict", {
      method: "POST",
      headers,
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();

    clearInterval(timer);
    // Navigate to results page
    const resultId = json.data.prediction_id || json.data.session_id;
    window.location.href = `/results/${resultId}`;
  } catch (err) {
    clearInterval(timer);
    overlay.style.display = "none";
    btn.disabled = false;
    alert("Prediction failed. Please try again.\n" + err.message);
  }
}
