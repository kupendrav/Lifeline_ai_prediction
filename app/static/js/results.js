/* LIFELINE AI — Results page animations */
'use strict';

document.addEventListener('DOMContentLoaded', function () {
  // Animate bars in when they enter viewport
  const bars = document.querySelectorAll('.ll-domain-bar-fill, .ll-shap-bar');
  bars.forEach(bar => {
    const targetWidth = bar.style.width;
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = targetWidth; }, 200);
  });

  // Animate result number count-up
  const bigNum = document.querySelector('.ll-result-big-num');
  if (bigNum) {
    const target = parseInt(bigNum.textContent);
    let current = Math.max(target - 20, 0);
    bigNum.textContent = current;
    const step = () => {
      if (current < target) {
        current = Math.min(target, current + 1);
        bigNum.textContent = current;
        requestAnimationFrame(step);
      }
    };
    setTimeout(step, 400);
  }
});
