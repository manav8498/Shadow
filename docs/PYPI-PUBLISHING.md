# PyPI publishing

Shadow publishes to PyPI via [**Trusted Publisher**](https://docs.pypi.org/trusted-publishers/)
— OpenID Connect from GitHub Actions to PyPI, no API token stored anywhere.
The workflow job in `.github/workflows/release.yml` (`publish-pypi`)
fires on every `v*` tag push and uploads whatever wheels the
`python-wheel` matrix built.

This document is the one-time setup. Once configured, every `git tag
vX.Y.Z && git push origin vX.Y.Z` auto-publishes.

## One-time setup (maintainer only)

### 1. Pre-register the project on PyPI

Before the first upload, PyPI needs to know which repo+workflow is
allowed to publish to the `shadow` name.

1. Log in to [pypi.org](https://pypi.org).
2. Go to **Your account → Publishing**
   (https://pypi.org/manage/account/publishing/).
3. Scroll to **"Add a new pending publisher"** and fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `shadow-diff` |
   | Owner | `manav8498` |
   | Repository name | `Shadow` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

This creates a *pending* trusted publisher. The first successful
publish from the matching workflow claims the `shadow-diff` name
and promotes the pending publisher to a permanent one.

> **Note on the name:** the PyPI project is `shadow-diff` because
> the bare `shadow` name was already registered on PyPI (an
> unrelated btrfs-snapshot utility from 2015). The Python import,
> CLI, and repo slug all stay `shadow`; only the `pip install`
> name differs.

### 2. Create the `pypi` GitHub Environment

The workflow binds the publish step to a GitHub Environment named
`pypi`. Creating the environment lets you add protection rules
(required reviewers, deployment branches) without modifying the
workflow.

1. In the Shadow repo, go to
   **Settings → Environments → New environment**.
2. Name it `pypi` (must match `environment.name` in `release.yml`).
3. Under **Deployment branches and tags**, select
   **Selected branches and tags** and add the rule `v*` so only
   version tags can trigger a publish.
4. (Optional, recommended for production) Add yourself as a
   **Required reviewer**. This adds a manual approval step before
   every publish — useful if you want to eyeball the release one more
   time before it hits PyPI.
5. Save.

### 3. Publish a tag

Once steps 1 and 2 are done, every tag push runs the publish job:

```bash
# Bump versions, commit, tag, push.
git tag -a v0.2.1 -m "shadow v0.2.1"
git push origin v0.2.1
```

Watch the **release** workflow run at
https://github.com/manav8498/Shadow/actions/workflows/release.yml.
If a required reviewer is configured, the `publish-pypi` job waits
for approval.

On success: `pip install shadow-diff==0.2.2` works within ~60 s.

## Backfilling v0.2.0

The v0.2.0 tag was pushed *before* this workflow step existed, so
the v0.2.0 wheels aren't on PyPI yet. Once the trusted publisher is
set up, re-run the release workflow for v0.2.0 from the Actions UI:

1. Go to **Actions → release → v0.2.0 run**.
2. Click **Re-run all jobs**.

The `publish-pypi` step will upload the v0.2.0 wheels that were
built and attached to the GitHub Release.

Alternatively (if re-running the whole workflow is overkill), run
`pypa/gh-action-pypi-publish` directly in a one-shot workflow
targeting v0.2.0.

## If the publish step fails

| Error | Cause | Fix |
|---|---|---|
| `403 Invalid or non-existent authentication` | Trusted publisher not configured | Go back to step 1 |
| `Environment protection rules not satisfied` | Reviewer approval pending | Approve the deployment in GitHub UI |
| `File already exists` | Version was already published | `skip-existing: true` already handles this; otherwise bump the version |
| `No matching wheels` | `python-wheel` job failed or produced nothing | Check the matrix job logs |

## Why Trusted Publisher (not an API token)?

- **No long-lived secret** to rotate, leak, or lose. OIDC tokens are
  short-lived and scoped to the exact (repo, workflow, environment)
  tuple the trusted publisher was registered against.
- **Deliberate trust boundary.** Reconfiguring any of those four
  fields on either side (repo, workflow name, environment name,
  PyPI config) breaks the trust link and requires a human to
  re-establish it. That's the point.
- **Better audit trail.** Every publish shows up on PyPI's activity
  log with the exact GitHub run ID that produced the upload.

PyPI [recommends trusted publishers](https://docs.pypi.org/trusted-publishers/)
over API tokens for new projects. Shadow follows that guidance.
