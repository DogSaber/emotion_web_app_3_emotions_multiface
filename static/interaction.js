(() => {
  'use strict';
  const selector = 'button:not([disabled]), a[href]:not([aria-disabled="true"])';
  const clear = () => document.querySelectorAll('.is-pressed').forEach(el => el.classList.remove('is-pressed'));
  document.addEventListener('pointerdown', event => event.target.closest(selector)?.classList.add('is-pressed'));
  document.addEventListener('pointerup', clear);
  document.addEventListener('pointercancel', clear);
  document.addEventListener('keydown', event => {
    if ((event.key === 'Enter' || event.key === ' ') && event.target.matches(selector)) event.target.classList.add('is-pressed');
  });
  document.addEventListener('keyup', clear);
})();
