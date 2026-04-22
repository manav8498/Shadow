"""Typer-based CLI. `shadow` console_script points at `shadow.cli.app.main`."""

from shadow.cli.app import app, main

__all__ = ["app", "main"]
