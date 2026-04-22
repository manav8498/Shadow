"""Report renderers for DiffReport dicts (terminal, markdown, github-pr)."""

from shadow.report.github_pr import render_github_pr
from shadow.report.markdown import render_markdown
from shadow.report.terminal import render_terminal

__all__ = ["render_github_pr", "render_markdown", "render_terminal"]
