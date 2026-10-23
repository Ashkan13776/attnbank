# Continuous integration

`ci.yml` is a ready-to-use GitHub Actions workflow running `pytest` and `ruff`
across the supported Python versions.

It is **not active yet**. It lives here rather than at `.github/workflows/`
because the token used to create this repository lacked the `workflow` OAuth
scope, and GitHub refuses pushes that add a workflow file without it.

To activate:

```bash
gh auth refresh -s workflow      # one-time, opens a browser
git mv ci/ci.yml .github/workflows/ci.yml
git commit -m "Activate CI" && git push
```
