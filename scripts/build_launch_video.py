#!/usr/bin/env python3
"""Build the Shadow product-launch video.

Pipeline (all artifacts under `.github/assets/`):

    1. Render the terminal recording with vhs (`launch.tape` →
       `launch-raw.mp4`).
    2. Generate the intro and outro card PNGs at 1920×1080 with
       PIL — clean dark background, Helvetica title, mauve accent.
    3. Synthesise an ambient chord-pad music bed via ffmpeg's
       sine generators (~80 s, fades in/out, low volume).
    4. Composite everything via ffmpeg: intro card → terminal →
       outro card with crossfades, vignette, and music ducked
       under the visual track.

The final output is ``.github/assets/launch.mp4`` (1920×1080 H.264 +
AAC). To swap in your own music, drop a file at
``.github/assets/launch-music.wav|.mp3`` and re-run this script — it
prefers a user-supplied music file over the synthesised bed when
present.

Run from repo root:

    PATH="$PWD/.venv/bin:$PATH" python scripts/build_launch_video.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / ".github" / "assets"
TAPE = ASSETS / "launch.tape"
RAW_TERMINAL = ASSETS / "launch-raw.mp4"
INTRO_PNG = ASSETS / "_launch-intro.png"
OUTRO_PNG = ASSETS / "_launch-outro.png"
MUSIC_WAV = ASSETS / "_launch-music.wav"
USER_MUSIC = [
    ASSETS / "launch-music.wav",
    ASSETS / "launch-music.mp3",
    ASSETS / "launch-music.m4a",
]
FINAL_MP4 = ASSETS / "launch.mp4"
FINAL_WEBM = ASSETS / "launch.webm"

# ─── Style ───────────────────────────────────────────────────────
ACCENT = (203, 166, 247)  # Catppuccin mauve
BG = (10, 10, 15)  # near-black, matches terminal margin
TITLE_RGB = (255, 255, 255)
SUBTITLE_RGB = (180, 180, 200)
DIM_RGB = (110, 110, 130)

# Times in seconds for the assembly.
INTRO_DURATION = 4.0
OUTRO_DURATION = 6.0
CROSSFADE = 0.6


def _font(size: int, *, bold: bool = False) -> object:
    """Pick a system font path; fall back through a small chain."""
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                # `index=1` typically picks the bold cut on .ttc files.
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def _font_mono(size: int) -> object:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _centred_text(draw, *, xy, text, font, fill):
    """Draw text horizontally centred at ``xy = (cx, y)``."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    cx, y = xy
    draw.text((cx - width // 2, y), text, font=font, fill=fill)


def build_intro_card() -> Path:
    """Generate the 4-second intro card.

    Centred logo · large title · subtitle · accent bar.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(img)

    # Subtle vignette gradient (radial darken at corners).
    overlay = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.ellipse((-400, -400, 2320, 1480), fill=(0, 0, 0, 0))
    odraw.ellipse((300, 200, 1620, 880), fill=(0, 0, 0, 0))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Accent bar — thin mauve line above the title.
    draw.rectangle((860, 380, 1060, 384), fill=ACCENT)

    # Title.
    title_font = _font(140, bold=True)
    _centred_text(
        draw,
        xy=(960, 420),
        text="Shadow",
        font=title_font,
        fill=TITLE_RGB,
    )

    # Subtitle.
    sub_font = _font(38)
    _centred_text(
        draw,
        xy=(960, 600),
        text="Behavior testing for LLM agents,",
        font=sub_font,
        fill=SUBTITLE_RGB,
    )
    _centred_text(
        draw,
        xy=(960, 656),
        text="in the pull request.",
        font=sub_font,
        fill=SUBTITLE_RGB,
    )

    # Footer — install command in mono.
    mono_font = _font_mono(28)
    _centred_text(
        draw,
        xy=(960, 870),
        text="$ pip install shadow-diff",
        font=mono_font,
        fill=DIM_RGB,
    )

    img.save(INTRO_PNG, optimize=True)
    return INTRO_PNG


def build_outro_card() -> Path:
    """Generate the outro card — CTA + repo URL."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(img)

    # Accent bar.
    draw.rectangle((860, 280, 1060, 284), fill=ACCENT)

    title_font = _font(110, bold=True)
    _centred_text(
        draw,
        xy=(960, 320),
        text="Try it now",
        font=title_font,
        fill=TITLE_RGB,
    )

    mono_font = _font_mono(40)
    _centred_text(
        draw,
        xy=(960, 520),
        text="$ pip install shadow-diff",
        font=mono_font,
        fill=ACCENT,
    )
    _centred_text(
        draw,
        xy=(960, 590),
        text="$ shadow demo",
        font=mono_font,
        fill=ACCENT,
    )

    sub_font = _font(34)
    _centred_text(
        draw,
        xy=(960, 800),
        text="github.com/manav8498/Shadow",
        font=sub_font,
        fill=SUBTITLE_RGB,
    )

    img.save(OUTRO_PNG, optimize=True)
    return OUTRO_PNG


def synth_music(duration: float) -> Path:
    """Generate an ambient chord pad via ffmpeg sine generators.

    Layers four sines at A2 / E3 / A3 / C#4 (Amaj triad voicing) at
    very low volume, with slow fades in/out. Sounds like a held
    cinematic chord — enough to add warmth without distracting from
    the terminal content.
    """
    # Amaj voicing: A2=110, E3=164.81, A3=220, C#4=277.18.
    freqs = [110.0, 164.81, 220.0, 277.18]
    inputs = []
    for f in freqs:
        inputs.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{duration:.2f}",
                "-i",
                f"sine=frequency={f}:sample_rate=44100",
            ]
        )

    # Mix the four tones, drop the volume hard, fade in/out, output WAV.
    n = len(freqs)
    filter_complex = (
        f"[0][1][2][3]amix=inputs={n}:duration=longest:dropout_transition=0[m];"
        f"[m]volume=0.10[v];"
        f"[v]afade=t=in:st=0:d=2.0,afade=t=out:st={duration - 2.0:.2f}:d=2.0[a]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[a]",
        "-c:a",
        "pcm_s16le",
        str(MUSIC_WAV),
    ]
    subprocess.run(cmd, check=True)
    return MUSIC_WAV


def music_source(target_duration: float) -> Path:
    """Return a music WAV/MP3/M4A path. Prefer a user-supplied file
    in `.github/assets/launch-music.*`; otherwise synthesise."""
    for candidate in USER_MUSIC:
        if candidate.is_file():
            print(f"  ↳ using user-supplied music: {candidate.name}")
            return candidate
    print("  ↳ synthesising ambient pad (drop launch-music.{wav,mp3} to override)")
    return synth_music(target_duration)


def render_terminal() -> Path:
    """Run vhs against `launch.tape` to produce the raw terminal MP4."""
    if not shutil.which("vhs"):
        raise SystemExit(
            "vhs not found on PATH. Install via `brew install vhs` "
            "or use the Docker fallback documented in "
            ".github/assets/README.md."
        )
    cmd = ["vhs", str(TAPE)]
    print(f"  ↳ vhs {TAPE.relative_to(REPO)}")
    subprocess.run(cmd, check=True, cwd=REPO)
    if not RAW_TERMINAL.is_file():
        raise SystemExit(f"expected {RAW_TERMINAL} not produced by vhs")
    return RAW_TERMINAL


def probe_duration(path: Path) -> float:
    """ffprobe the duration of a media file (seconds)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def composite(
    *,
    terminal: Path,
    intro: Path,
    outro: Path,
    music: Path,
) -> Path:
    """Composite intro card + terminal + outro card with crossfades,
    vignette, and the music bed.

    The terminal recording's audio (silent) is dropped; only the music
    track survives. A subtle vignette is applied to the visual stream
    over the entire timeline."""
    term_dur = probe_duration(terminal)
    total = INTRO_DURATION + term_dur + OUTRO_DURATION - 2 * CROSSFADE
    print(
        f"  ↳ pieces: intro {INTRO_DURATION}s + terminal {term_dur:.1f}s "
        f"+ outro {OUTRO_DURATION}s, total ~{total:.1f}s"
    )

    # Build a single ffmpeg pipeline that:
    #   - Loops the intro PNG for INTRO_DURATION seconds
    #   - Loops the outro PNG for OUTRO_DURATION seconds
    #   - Reads the terminal MP4
    #   - xfades intro -> terminal -> outro
    #   - Applies vignette
    #   - Mixes the music track at low volume
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-stats",
        # Inputs ----------------------------------------------------
        "-loop",
        "1",
        "-t",
        f"{INTRO_DURATION}",
        "-i",
        str(intro),
        "-i",
        str(terminal),
        "-loop",
        "1",
        "-t",
        f"{OUTRO_DURATION}",
        "-i",
        str(outro),
        "-i",
        str(music),
        # Filter graph ---------------------------------------------
        "-filter_complex",
        (
            # Match intro/outro to terminal's pixel format + size.
            f"[0:v]scale=1920:1080,format=yuv420p,setsar=1,fps=30[intro];"
            f"[1:v]scale=1920:1080,format=yuv420p,setsar=1,fps=30[term];"
            f"[2:v]scale=1920:1080,format=yuv420p,setsar=1,fps=30[outro];"
            # Crossfades.
            f"[intro][term]xfade=transition=fade:duration={CROSSFADE}:"
            f"offset={INTRO_DURATION - CROSSFADE}[ab];"
            f"[ab][outro]xfade=transition=fade:duration={CROSSFADE}:"
            f"offset={INTRO_DURATION + term_dur - 2 * CROSSFADE}[v];"
            # Subtle vignette over the whole thing.
            f"[v]vignette=PI/5[vv];"
            # Music — trim/loop to fit the visual length and fade out.
            f"[3:a]aloop=loop=-1:size=2e+09,atrim=duration={total:.2f},"
            f"afade=t=out:st={total - 1.5:.2f}:d=1.5[a]"
        ),
        "-map",
        "[vv]",
        "-map",
        "[a]",
        # Encode --------------------------------------------------
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(FINAL_MP4),
    ]
    subprocess.run(cmd, check=True)
    print(f"  ↳ wrote {FINAL_MP4.relative_to(REPO)} ({FINAL_MP4.stat().st_size // 1024} KB)")
    return FINAL_MP4


def encode_webm(source: Path) -> Path:
    """Encode a VP9 + Opus WebM from the final MP4 for web embed."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "32",
        "-b:v",
        "0",
        "-c:a",
        "libopus",
        "-b:a",
        "128k",
        str(FINAL_WEBM),
    ]
    subprocess.run(cmd, check=True)
    return FINAL_WEBM


def main() -> None:
    print("Shadow launch-video build")
    print("─" * 50)

    print("[1/5] Rendering terminal recording (vhs)…")
    terminal = render_terminal()

    print("[2/5] Generating intro / outro cards…")
    intro = build_intro_card()
    outro = build_outro_card()
    print(f"  ↳ {intro.relative_to(REPO)}, {outro.relative_to(REPO)}")

    print("[3/5] Generating music bed…")
    term_dur = probe_duration(terminal)
    target = INTRO_DURATION + term_dur + OUTRO_DURATION - 2 * CROSSFADE
    music = music_source(target_duration=target + 1.0)

    print("[4/5] Compositing…")
    composite(terminal=terminal, intro=intro, outro=outro, music=music)

    print("[5/5] Encoding WebM…")
    encode_webm(FINAL_MP4)
    print(f"  ↳ wrote {FINAL_WEBM.relative_to(REPO)} "
          f"({FINAL_WEBM.stat().st_size // 1024} KB)")

    print()
    print("Done.")
    print(f"  {FINAL_MP4.relative_to(REPO)}  (H.264 + AAC, post to X / LinkedIn)")
    print(f"  {FINAL_WEBM.relative_to(REPO)}  (VP9 + Opus, for web embed)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"command failed: {' '.join(str(x) for x in e.cmd)}", file=sys.stderr)
        raise SystemExit(e.returncode) from e
