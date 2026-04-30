# Shadow assets

Static images and videos embedded in `README.md`.

| File | Source | Notes |
|---|---|---|
| `logo.png` | hand-drawn | Used as the README header. |
| `demo.gif` | rendered from `demo.tape` | Workflow-loop walkthrough. ~860 KB, 1400 × 900, 59 s, animated GIF (auto-plays). |
| `demo.mp4` | rendered from `demo.tape` | Same content, H.264 video. ~600 KB, 1400 × 900, 25 fps, 59 s. |
| `demo.webm` | rendered from `demo.tape` | Same content, VP9 video. ~700 KB, 1400 × 900, 59 s. |
| `demo.tape` | hand-written | [vhs](https://github.com/charmbracelet/vhs) script. |

## Regenerating the demo assets

All three demo files (`demo.gif`, `demo.mp4`, `demo.webm`) are rendered
from a single vhs run over `demo.tape`. To re-render after editing:

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
all three artifacts alongside the source change. The tape is the
source of truth; the GIF / MP4 / WebM are derivatives.
