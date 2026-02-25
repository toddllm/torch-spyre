(function () {
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  window.Prism = {
    // Escape-only highlighter to preserve exact code text fidelity.
    highlight: function (code) {
      return escapeHtml(code);
    },
    languages: {}
  };
})();
