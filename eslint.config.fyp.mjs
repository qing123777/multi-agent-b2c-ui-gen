// Static lint config used by the FYP notebook's run_eslint tool.
// Browser globals declared read-only: generated code is front-end only.
export default [
  {
    languageOptions: {
      ecmaVersion: "latest",
      globals: {
        window: "readonly", document: "readonly", console: "readonly",
        localStorage: "readonly", sessionStorage: "readonly", fetch: "readonly",
        alert: "readonly", setTimeout: "readonly", setInterval: "readonly",
        clearTimeout: "readonly", clearInterval: "readonly",
        navigator: "readonly", location: "readonly", history: "readonly",
        URLSearchParams: "readonly", Event: "readonly", CustomEvent: "readonly",
        FormData: "readonly", IntersectionObserver: "readonly",
        requestAnimationFrame: "readonly"
      }
    },
    rules: { "no-undef": "error", "no-unused-vars": "warn" }
  }
];
