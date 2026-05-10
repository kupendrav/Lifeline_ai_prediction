/* LIFELINE AI — Form interactions */

'use strict';

function updateSlider(id, val, suffix) {
  const out = document.getElementById(id + '-out');
  if (out) out.textContent = val + suffix;
}

function updateBMI(val) {
  const num = parseFloat(val);
  document.getElementById('bmi-out').textContent = num.toFixed(1);
  const label = document.getElementById('bmi-label');
  let text = '', color = '';
  if (num < 18.5)      { text = 'Underweight';    color = '#B8930A'; }
  else if (num < 25)   { text = 'Normal weight';   color = '#2D6E6A'; }
  else if (num < 30)   { text = 'Overweight';      color = '#B8930A'; }
  else if (num < 35)   { text = 'Obese (class I)'; color = '#C0543A'; }
  else if (num < 40)   { text = 'Obese (class II)';color = '#C0543A'; }
  else                 { text = 'Severe obesity';   color = '#8B1A1A'; }
  label.textContent = text;
  label.style.color = color;
  label.style.background = color + '18';
}

function updateBP(val) {
  const num = parseInt(val);
  document.getElementById('blood_pressure-out').textContent = num + ' mmHg';
  const label = document.getElementById('bp-label');
  let text = '', color = '';
  if (num < 120)       { text = 'Optimal';     color = '#2D6E6A'; }
  else if (num < 130)  { text = 'Elevated';    color = '#B8930A'; }
  else if (num < 140)  { text = 'Stage 1 HTN'; color = '#B8930A'; }
  else if (num < 160)  { text = 'Stage 2 HTN'; color = '#C0543A'; }
  else                 { text = 'Hypertensive crisis'; color = '#8B1A1A'; }
  label.textContent = text;
  label.style.color = color;
  label.style.background = color + '18';
}

function updateAQI(val) {
  const num = parseInt(val);
  document.getElementById('aqi-out').textContent = num;
  const label = document.getElementById('aqi-label');
  let text = '', color = '';
  if (num <= 50)       { text = 'Good';             color = '#2D6E6A'; }
  else if (num <= 100) { text = 'Moderate';          color = '#B8930A'; }
  else if (num <= 150) { text = 'Unhealthy for sensitive groups'; color = '#C0543A'; }
  else if (num <= 200) { text = 'Unhealthy';         color = '#C0543A'; }
  else if (num <= 300) { text = 'Very Unhealthy';    color = '#8B1A1A'; }
  else                 { text = 'Hazardous';          color = '#6B0000'; }
  label.textContent = text;
  label.style.color = color;
  label.style.background = color + '18';
}

// Form submit: show loading state
document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('longevity-form');
  const btn  = document.getElementById('submit-btn');

  if (form && btn) {
    form.addEventListener('submit', function () {
      btn.disabled = true;
      btn.innerHTML = '<i class="ti ti-loader-2" style="animation:spin 1s linear infinite"></i> Analysing...';
    });
  }

  // Initialise labels on page load
  const bmiEl = document.getElementById('bmi');
  const bpEl  = document.getElementById('blood_pressure');
  const aqiEl = document.getElementById('aqi');
  if (bmiEl) updateBMI(bmiEl.value);
  if (bpEl)  updateBP(bpEl.value);
  if (aqiEl) updateAQI(aqiEl.value);
});

// Add spin animation for loading icon
const spinStyle = document.createElement('style');
spinStyle.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
document.head.appendChild(spinStyle);
