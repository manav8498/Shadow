# Shadow assets

Static images and videos embedded in `README.md`.

| File | Source | Notes |
|---|---|---|
| `logo.png` | hand-drawn | Used as the README header. |
| `demo.gif` | rendered from `demo.tape` | Workflow-loop walkthrough. ~860 KB, 1400 × 900, 59 s, animated GIF (auto-plays). |
| `demo.mp4` | rendered from `demo.tape` | Same content, H.264 video. ~600 KB, 1400 × 900, 25 fps, 59 s. |
| `demo.webm` | rendered from `demo.tape` | Same content, VP9 video. ~700 KB, 1400 × 900, 59 s. |
| `demo.tape` | hand-written | [vhs](https://github.com/charmbracelet/vhs) script. |
| `launch.mp4` | built by `scripts/build_launch_video.py` | Launch-video cut for X / LinkedIn. ~4.3 MB, 1920 × 1080, 30 fps, 84 s, H.264 + AAC. Six-beat tour (`demo`, `call`, `autopr`, `ledger`, `trail`, `certify`) with intro / outro cards and ambient music bed. |
| `launch.webm` | built by `scripts/build_launch_video.py` | Same content, VP9 + Opus. ~3.0 MB, 1920 × 1080, 84 s. |
| `launch.tape` | hand-written | vhs script — the terminal-only portion (~75 s). |

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

## Regenerating the launch video

`launch.mp4` / `launch.webm` are produced by a single Python
orchestrator that wraps the vhs render with intro / outro cards,
an ambient music bed, and crossfades:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/build_launch_video.py
```

The script renders `launch.tape` via vhs, generates the title
cards with PIL, synthesises an ambient pad with ffmpeg (drop
`launch-music.{wav,mp3,m4a}` next to the tape to override),
composites everything into `launch.mp4`, and re-encodes
`launch.webm`. Intermediates (`_launch-*.png`, `_launch-*.wav`,
`launch-raw.*`) are gitignored — only `launch.mp4` /
`launch.webm` are committed.
