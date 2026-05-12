# Locked install (byte-identical environments)

The default `pip install shadow-diff` resolves transitive dependencies fresh against your existing environment, which is what you want for soft coexistence with an agent stack that already pins (e.g. LangChain pre-0.3 pinning `numpy<2`). For CI repro, enterprise audit, and debugging "works on my machine" reports, Shadow also publishes a fully pinned dependency set that resolves to byte-identical packages on every install.

## When to use it

- CI jobs that must reproduce a known-good resolution across reruns.
- Enterprise audit / vendoring workflows that diff dependency trees release-to-release.
- Bisecting a regression between two Shadow versions without ambient resolver drift.

## Install paths

`uv` users (recommended):

```bash
uv sync --frozen --project python
```

`pip` / `pip-tools` users:

```bash
pip install -r requirements-locked.txt
pip install --no-deps shadow-diff
```

The `--no-deps` follow-up installs Shadow itself without re-resolving — every transitive dep already came from `requirements-locked.txt`.

## Caveats

Locked install **will fail** if your existing project pins packages that conflict with Shadow's locked versions — most commonly `numpy<2` (LangChain pre-0.3), `pydantic<2` (older agent stacks), or `httpx<0.27`. If you see a resolver error, use the default `pip install shadow-diff` path instead; it picks permissive ranges so Shadow coexists with whatever your project already has installed.

The locked set pulls every optional extra (`[all]` plus `[dev]` test deps), which includes `sentence-transformers` (~2GB on first download). For a leaner locked install, run `uv export --extra <name>` against the lockfile yourself.

## When to regenerate

The lockfile is regenerated on every Shadow release. Maintainers run `uv lock --upgrade` against `python/pyproject.toml` and re-export `requirements-locked.txt` before tagging. A GitHub Actions job (`lockfile-sync-check`) fails the build if `uv.lock` drifts from `pyproject.toml` on `main`.

If you're pulling from `main` between releases and want the freshest pins, run `uv lock --upgrade --project python` locally — the same command CI uses.
