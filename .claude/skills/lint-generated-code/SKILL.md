---
name: lint-generated-code
description: Use when running, debugging, or modifying the linting tools (run_html_validate, run_eslint, run_stylelint) that gate the HTML/CSS/JS developer agents' self-correction loop. Includes hard-won gotchas from live debugging that mocked tests did not catch.
---

# Linting the generated frontend output

Each developer agent (HTML/CSS/JS) loops generate → lint → fix until its linter reports
zero errors, then writes the final file. The three linter tools live in `fyp.ipynb`
(locate by content — see [`notebook-editing`](../notebook-editing/SKILL.md)) and shell
out to Node CLIs via `npx`.

## Config files (repo root)

- `eslint.config.fyp.mjs` — flat config (ESLint v9+ format). Declares browser globals
  read-only (`window`, `document`, `fetch`, etc. — generated code is front-end only).
  Rules: `no-undef: error`, `no-unused-vars: warn`.
- `.stylelintrc.json` — 8 built-in rules (`block-no-empty`, `color-no-invalid-hex`,
  `property-no-unknown`, `selector-pseudo-class-no-unknown`, `selector-type-no-unknown`,
  `unit-no-unknown`, `no-duplicate-selectors`,
  `declaration-block-no-duplicate-properties`). No plugin dependencies.
- `html-validate` — no repo-local config file; invoked with defaults via `npx`.

## Running a linter manually

Always run from the repo root — see the `%TEMP%` gotcha below.

```bash
npx html-validate static/preview/preview_<sid>.html
npx eslint --config eslint.config.fyp.mjs static/preview/preview_<sid>.js
npx stylelint --config .stylelintrc.json static/preview/preview_<sid>.css
```

## Gate rule: lint_passed gates on errors, not warnings

`run_eslint`'s `lint_passed` flag is set from ESLint **errors only**. `no-unused-vars`
warnings are reported to the agent as advisory but never block the fix loop — this is
deliberate, a cosmetic unused-var warning must never trap the JS agent in an infinite
retry.

## Gotchas found by live testing (not caught by mocks)

These were all discovered by actually invoking the real binaries — offline mocked tests
had been green the whole time and gave no signal that the tools were broken.

- **`run_eslint` had never worked live.** The repo uses ESLint v10 (flat config), but
  the tool was calling it with flags removed since v9 (`--no-eslintrc`, inline
  `--rule`). Every invocation exited with a CLI usage error, which the loop
  misinterpreted as a normal lint failure. Fix: point at `eslint.config.fyp.mjs`
  explicitly, drop the removed flags.
- **ESLint v9+ silently ignores files outside its base path.** If a target file (or the
  temp copy used for linting) lives outside the directory ESLint considers its root, it
  is skipped with no error — looks like a clean pass when nothing was actually linted.
- **`npx` run from `%TEMP%` downloads a throwaway linter** instead of resolving the
  repo's local `node_modules` install. Always invoke `npx` with the working directory
  set to the repo root, and keep any per-run temp/scratch files in the repo-local
  `.lint_tmp/` directory (gitignored), never under the OS temp dir.
- **Stylelint v16+/v17 writes its JSON report to STDERR, not stdout.** A parser that
  only reads stdout sees empty output and reports "unparseable" — always parse
  `stdout.strip() or stderr.strip()`. Stylelint also exits with code 78 specifically for
  a config-resolution error; treat that distinctly from a normal lint-failure exit code
  (report it as an infra error, not a code-quality failure the agent should try to fix).
- **Stylelint requires a config file to run at all** — `.stylelintrc.json` must exist
  and be passed with `--config`.

## Where coverage validation fits (separate from linting)

Before a linter ever runs, `store_html_code`/`store_css_code`/`store_js_code` each run a
**coverage validator** that rejects incomplete submissions outright (missing
`<link>`/`<script>` includes, a spec component with no addressable HTML, a decoration
`className` with no CSS rule, an interaction `targetId` never referenced in the JS).
This is a separate, code-enforced guard layer — don't confuse a coverage-validator
rejection with a lint failure when debugging a stuck fix loop; check which layer is
actually rejecting the submission first.

## Debugging a stuck agent lint loop

1. Reproduce the exact file the agent last wrote (from `static/preview/` or from state)
   and run the linter manually with the commands above from the repo root.
2. If the manual run passes but the agent loop still reports failure, suspect a
   working-directory or temp-file issue (see gotchas above) before assuming the
   generated code is actually broken.
3. If the manual run fails, read the raw linter output (stdout **and** stderr) rather
   than trusting the tool's parsed summary — confirms the parser itself isn't
   misreading a valid report.
