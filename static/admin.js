(() => {
  const button = document.getElementById('adminMenuBtn');
  const sidebar = document.getElementById('adminSidebar');
  const backdrop = document.getElementById('adminSidebarBackdrop');

  if (!button || !sidebar || !backdrop) {
    return;
  }

  const closeSidebar = () => {
    sidebar.classList.remove('open');
    backdrop.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  };

  button.addEventListener('click', () => {
    const isOpen = !sidebar.classList.contains('open');
    sidebar.classList.toggle('open', isOpen);
    backdrop.hidden = !isOpen;
    button.setAttribute('aria-expanded', String(isOpen));
  });

  backdrop.addEventListener('click', closeSidebar);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeSidebar();
    }
  });
})();
