"""Resource directory for `shadow quickstart`.

Shipped as part of the wheel so `importlib.resources` can stream
the bundled agent + configs + pre-recorded `.agentlog` fixtures
out into a user's working directory on `shadow quickstart`.

Not importable user-facing helpers — everything here is data. If
you need a runtime helper, put it in `shadow.cli.quickstart`
instead.
"""
