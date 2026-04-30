# Releasing Shadow

This is the practical guide for cutting a release. Aimed at the
single maintainer + occasional contributors who land PRs through
the normal review flow. If you're a contributor wondering how to
get a change shipped, you don't need to read this file — just open
a PR with a [Conventional Commits](https://www.conventionalcommits.org/)
subject (`feat:`, `fix:`, `docs:`, …) and the maintainer takes it
from there.

## TL;DR — the normal release

Releases are **automated** via [release-please](https://github.com/googleapis/release-please).
Every push to `main` updates a long-running PR titled
`chore: release X.Y.Z`. Merging that PR creates the `vX.Y.Z` git
tag and a GitHub Release; the existing `release.yml` workflow then
publishes signed artifacts to PyPI, npm, and crates.io.

So the maintainer flow for a normal release is:

1. Land changes through PRs as usual. Subjects must follow
   Conventional Commits — `feat:` triggers a minor bump, `fix:` /
   `perf:` a patch, anything with `!` after the type or a
   `BREAKING CHANGE:` footer a major.
2. When the release-please PR shows the right next version,
   approve and merge it.
3. The publish workflow fires automatically. Watch
   [Actions → release](https://github.com/manav8498/Shadow/actions/workflows/release.yml)
   for the result.
4. Confirm the new version on each registry:
   - https://pypi.org/project/shadow-diff/
   - https://www.npmjs.com/package/shadow-diff
   - https://crates.io/crates/shadow-diff

That's it. Everything below is for the cases where step 3 doesn't
quite work cleanly (missing secret, partial upload, you want a
specific version on the next release, etc.).

## One-time setup (per fresh repo)

Three repository secrets gate the publish steps. Without them, the
release workflow runs and emits a loud `::warning` per missing
secret, but no actual publish happens. Set them once and forget.

Add at https://github.com/manav8498/Shadow/settings/secrets/actions
→ "New repository secret":

| Secret | Where to get it | Scope |
|---|---|---|
| `CARGO_REGISTRY_TOKEN` | https://crates.io/settings/tokens → "New Token" | `publish-new` + `publish-update` on the `shadow-diff` crate |
| `NPM_TOKEN` | https://www.npmjs.com/settings/manav8498/tokens → "Generate New Token" → "Granular Access Token" | Read + Write on the `shadow-diff` package |

PyPI uses **trusted publishing** (OIDC) and does not need a stored
token. The trust link is registered against
`(manav8498, Shadow, release.yml, pypi environment)` — changing any
of those four fields breaks the trust link on purpose. If you fork
the repo or rename the workflow, re-register at
https://pypi.org/manage/account/publishing/.

## Manual override: force a specific version

If you need to skip ahead (e.g. release-please thinks the next
version is `3.1.0` but you want `4.0.0` for marketing reasons),
land an empty commit on `main` with a `Release-As:` footer:

```bash
git commit --allow-empty -s -m "chore: bump to 4.0.0

Release-As: 4.0.0"
git push origin main
```

The next release-please run will produce a PR for `4.0.0` instead
of whatever it would have computed. Merge that PR to release.

## Manual override: cut a release without release-please

Sometimes release-please gets stuck (`untagged, merged release PRs
outstanding` error after a force-push to its branch, for example).
Cut the release by hand:

```bash
# Bump the version files (the same ones release-please would touch)
# - .release-please-manifest.json
# - Cargo.toml (workspace.package.version)
# - python/pyproject.toml ([project] version)
# - python/src/shadow/__init__.py (__version__)
# - typescript/package.json
# - typescript/package-lock.json (root + packages."" version)
# - README.md badge URL

# Add a CHANGELOG.md entry under [Unreleased] -> [X.Y.Z]

git commit -s -am "chore: release X.Y.Z"
git push origin main
git tag -a vX.Y.Z -m "release X.Y.Z"
git push origin vX.Y.Z
```

The `release.yml` workflow triggers on `push: tags: v*`, so the
tag push alone fires the publish chain without needing release-
please.

## Troubleshooting

### `400 License-File X does not exist in distribution` on PyPI

PyPI's PEP 639 validator (strict since 2025) rejects wheels whose
METADATA declares a `License-File:` entry that the wheel itself
doesn't contain. Cause: maturin auto-detects every `LICENSE*` file
at the project root and adds it to METADATA, but the actual file
must be packaged into the wheel.

Fix: in `python/pyproject.toml`, set `license-files` explicitly to
the files you intend to package:

```toml
[project]
license = "Apache-2.0"
license-files = ["LICENSE-APACHE"]
```

Maturin honours `license-files` for both sdist and wheel.

### `error: crate name 'X' is already taken` on crates.io

Someone else owns the crate name. Either pick a different name in
`crates/shadow-core/Cargo.toml`'s `[package] name` (the Rust
import path stays unchanged because `[lib] name = "shadow_core"`
is set independently), or contact crates.io support to argue
prior art.

### `npm ERR! 403 Forbidden — You do not have permission to publish "X"`

The package name is taken or you don't own the scope. For an
unscoped name, pick something else. For a scoped name like
`@shadow/sdk`, you need to first create the `@shadow` organization
on https://www.npmjs.com/org/create.

### `Swatinem/rust-cache@v2` failure during job setup on Windows

This is a known infrastructure flake on the GitHub-hosted Windows
runner. Re-run the failed jobs:

```bash
gh run rerun <run-id> --failed
```

Adjacent Windows Python jobs on the same commit will tell you
whether it's a real failure (all fail consistently) or a flake
(only one failed, sibling jobs passed).

### release-please aborts with "untagged, merged release PRs outstanding"

The release-please action tracks a `autorelease: pending` label on
release PRs. If you force-push to the release branch, the label
state can get out of sync — release-please refuses to open new
PRs while it thinks there's an unfinished one.

Fix: tag the merged PR's commit by hand (`git tag -a vX.Y.Z &&
git push origin vX.Y.Z`). The next release-please run will see
the tag exists and proceed normally.

## Yanking a partial / broken release

PyPI and npm support **yanking** — the release stays available
to anyone who pinned it explicitly, but new resolves skip it.
Use yank when an upload was incomplete or contains a known bug,
not for security advisories (those need the GitHub Security
Advisory flow).

### PyPI

1. https://pypi.org/manage/project/shadow-diff/release/X.Y.Z/
2. "Options" → "Yank release"
3. Provide a yank reason (visible to anyone who pins X.Y.Z)

Or via `twine`:

```bash
# Requires PyPI API token with "Maintainer" scope on shadow-diff.
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... \
  twine yank --version X.Y.Z shadow-diff --reason "incomplete upload"
```

### npm

```bash
npm deprecate shadow-diff@X.Y.Z "incomplete upload — use X.Y.Z+1"
```

(npm has a 72-hour `npm unpublish` window after publish; past
that, deprecate is the only option.)

### crates.io

```bash
cargo yank --version X.Y.Z shadow-diff
```

Crates.io yanks are reversible (`cargo yank --undo`).

## Checklist for a clean release

Run through this if you're about to merge the release-please PR
or push a manual tag:

- [ ] CI is green on `main` (all matrix jobs, not just the recent
      ones)
- [ ] CHANGELOG entry reads sensibly — release-please groups
      commits by Conventional Commit type, but you can edit the
      release-please PR's CHANGELOG section if you want curated
      wording
- [ ] Version is consistent across all version-bearing files
      (release-please handles this; if cutting manually, double-
      check the seven listed under "Manual override")
- [ ] No untagged work-in-progress on `main` that would slip into
      the release
- [ ] Stale `demo-full.*` / `_launch-*.{png,wav}` / `launch-raw.*`
      files in `.github/assets/` are not staged (they're gitignored
      by default; just check `git status` before the merge)
- [ ] `pip install shadow-diff==X.Y.Z` works in a clean venv after
      the publish completes (sanity check, takes 30 seconds)

If any item fails, prefer rolling forward with X.Y.Z+1 over
trying to fix X.Y.Z in place — registry semantics make
mid-version corrections expensive.
