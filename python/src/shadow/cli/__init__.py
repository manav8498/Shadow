"""Typer-based CLI.

The `shadow` console script points at `shadow.cli.app:main` directly
(see pyproject.toml `[project.scripts]`), so this package's
`__init__` stays empty. Eagerly re-exporting `app`/`main` here would
trigger a `RuntimeWarning` on Windows when callers run
`python -m shadow.cli.app` — the package loader imports
`shadow.cli.app` transitively, then runpy re-executes it, and Windows'
stricter warning reporting breaks any test that inspects stderr.

Import `shadow.cli.app` directly when you need the Typer object.
"""
