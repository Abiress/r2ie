# Contributing to R²IE

Thanks for your interest in contributing! This is a small research/education prototype,
so the bar is: keep it honest, keep it tested.

## How to contribute

1. Fork the repository and create a feature branch (`git checkout -b my-feature`).
2. Make your change. Add or update tests that exercise real tensor math / gradient flow.
3. Run the test suite locally:
   ```bash
   pip install -e ".[dev]"
   pytest -v
   ruff check src tests
   ```
4. Make sure **all tests pass** and **ruff is clean**. CI will reject PRs that fail either.
5. Open a pull request describing what changed and why.

## Rules

- No test may be written to trivially pass (e.g. `assert True`). Tests must exercise
  real behavior.
- No unverified performance or benchmark claims. If you ran a benchmark, paste its
  literal output into `BENCHMARKS.md` with the run command and hardware noted.
- Keep all authorship/credit as "Abir Maheshwari". Do not add agent- or AI-generated
  attribution.

## Reporting issues

Open an issue describing the expected behavior, the actual behavior, and a minimal
reproduction.
