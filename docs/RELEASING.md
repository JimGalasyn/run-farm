# Releasing

1. Bump `version` in `pyproject.toml`, `src/run_farm/__init__.py`, and
   `CITATION.cff` (keep them in sync).
2. Commit, tag `vX.Y.Z`, push the tag.
3. Publish a GitHub Release for the tag — this triggers
   `.github/workflows/publish-pypi.yml` (OIDC trusted publishing to PyPI).

## First publish (one-time)

- Add a **pending publisher** on pypi.org (Project `run-farm`, Owner
  `JimGalasyn`, Repo `run-farm`, Workflow `publish-pypi.yml`, Environment
  `pypi`), and create a GitHub Environment named `pypi`.
- For the DOI badge, connect the repo to Zenodo before the first release so it
  mints a concept DOI; add it to `README.md`, `CITATION.cff`, `.zenodo.json`.
- Add the repo to Codecov and set the `CODECOV_TOKEN` secret.
