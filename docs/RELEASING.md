# Releasing

> **Cross-repo release failure modes live in the `publish-release` skill**
> (`claude-shared/skills/publish-release`). It carries what has bitten across all of
> these repos — green workflows that published nothing, PyPI propagation lying in both
> directions, the DOI commit that the *next* release's changelog check always flags, and
> the one-time setup steps that cannot be undone. **This file keeps what is specific to
> this repo**; the two are deliberately not copies of each other, because duplicated
> process drifts.

Three steps do the release. The rest of this file is the things that have actually
gone wrong around them, each written next to the release it went wrong on — because
none of them were caught by a test, and every one was found later by someone
reading a file that disagreed with reality.

## Before tagging

1. **The CHANGELOG must describe every commit since the last tag.** Check it, do
   not remember it:

   ```bash
   git log --oneline $(git describe --tags --abbrev=0)..main
   ```

   At v0.4.0 this turned up **four of seven commits undocumented**, two of which
   had changed the public surface (`CapClearsWorstCase` became an export,
   `HostSpec` gained `min_gpu_ram_mb`). Tagging there would have shipped a version
   whose changelog omitted half of what was in it — the failure 0.3.0's own entry
   is a long meditation on. Write entries from the **commit messages**, not the
   diffs: this project's commits already record the campaign that paid for each
   lesson, and that context does not survive being re-derived from a diff.

2. **Bump `version` in `pyproject.toml` and `CITATION.cff`, and keep them in sync.**
   `src/run_farm/__init__.py` carries no `__version__` — do not add one without
   also wiring it into this list. `pyproject.toml`'s own comment explains the
   `.devN` convention and why the number is the diagnosis rather than the disguise;
   a release drops the suffix.

3. **Update the README status line.** It is the *only* hardcoded version in the
   README — everything around it (Release, PyPI, Python badges) is dynamic and
   stays correct on its own, which is exactly why this one drifted through **two
   releases** unnoticed, still reading `0.2.x` at 0.4.0.

## Tag and publish

4. Commit, then `git tag -a vX.Y.Z` (annotated, matching the existing tags), and
   push the tag.
5. Publish a GitHub Release for the tag. That triggers two things you did not run:
   `.github/workflows/publish-pypi.yml` (OIDC trusted publishing to PyPI) and,
   because the repo is connected to Zenodo, the minting of a **version DOI**.

## After publishing — the step that has been missed every time

6. **Record the new version DOI in `CITATION.cff`.** v0.1.1, v0.2.0 *and* v0.3.0
   were each minted and left unrecorded; v0.3.0's was recovered two releases late,
   during the 0.4.0 release. The recovery works, but it is a recovery, and a
   `CITATION.cff` that omits the DOI for the version it names is wrong in the one
   file whose entire job is to be cited correctly.

   Get the DOI by **querying Zenodo against the concept record** — never guess it
   from the numbering, which is not sequential:

   ```bash
   curl -s "https://zenodo.org/api/records?q=conceptdoi:%2210.5281/zenodo.21419776%22&all_versions=true&size=20" \
     | python3 -c "import json,sys; [print(h['metadata'].get('version'), h['metadata'].get('doi')) for h in json.load(sys.stdin)['hits']['hits']]"
   ```

   Set `date-released` to Zenodo's `publication_date`, not to your local date. A
   tag cut in the evening in North America is the next day in UTC — v0.4.0 was cut
   on the 14th locally and published on the 15th — and a citation file that
   disagrees with its own DOI's landing page ends up in someone's bibliography.

## Verify the artifacts, not the invocations

A green workflow means the workflow ran. It does not mean the package is on PyPI
or that the DOI resolves. Check the three things that are supposed to now exist:

```bash
gh run list --workflow publish-pypi.yml --limit 1        # did it run, and pass
curl -s https://pypi.org/pypi/run-farm/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
curl -s -o /dev/null -w "%{http_code}\n" -L https://doi.org/<the new DOI>
```

The DOI check matters more than it looks: the Zenodo *listing* will show a record
before you have confirmed it resolves, and asserting a DOI from a listing is the
same mistake as asserting a file is good from a directory listing.

## First publish (one-time)

- Add a **pending publisher** on pypi.org (Project `run-farm`, Owner
  `JimGalasyn`, Repo `run-farm`, Workflow `publish-pypi.yml`, Environment
  `pypi`), and create a GitHub Environment named `pypi`.
- For the DOI badge, connect the repo to Zenodo before the first release so it
  mints a concept DOI; add it to `README.md`, `CITATION.cff`, `.zenodo.json`.
- Add the repo to Codecov and set the `CODECOV_TOKEN` secret.
