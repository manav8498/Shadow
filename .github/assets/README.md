# Shadow assets

Static images embedded in `README.md`.

| File | Source | Notes |
|---|---|---|
| `logo.png` | hand-drawn | Used as the README header. |
| `demo.gif` | rendered from `demo.tape` | Workflow-loop walkthrough. ~1 MB, 1400 × 900, 59 s. |
| `demo.tape` | hand-written | [vhs](https://github.com/charmbracelet/vhs) script. |

## Regenerating `demo.gif`

The GIF is a vhs render of `demo.tape`. To re-render after editing:

```bash
# Local install (macOS)
brew install vhs
PATH="$PWD/.venv/bin:$PATH" vhs .github/assets/demo.tape

# Or via Docker (no local install required)
docker run --rm -v "$PWD:/vhs" ghcr.io/charmbracelet/vhs \
    .github/assets/demo.tape
```

The tape assumes `shadow` is on `PATH` and the bundled fixtures
(`shadow quickstart`) are reachable. Both come from a standard
`pip install shadow-diff` followed by `maturin develop` in this repo.

If you change a command's output formatting, re-render and commit
the resulting `demo.gif` alongside the source change. The tape is
the source of truth; the GIF is the artifact.
