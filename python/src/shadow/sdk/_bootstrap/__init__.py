"""Bootstrap directory.

This subpackage's *directory* is prepended to the child process's
PYTHONPATH by `shadow record`, causing Python to find our
`sitecustomize.py` first on interpreter startup. The `__init__.py`
itself is just a locator so `Path(_bootstrap.__file__).parent`
works regardless of where shadow was installed.

Do not import user-facing helpers from this package; add them to
`shadow.sdk` and re-export from there. Anything importable here
lives behind the sitecustomize gate.
"""
