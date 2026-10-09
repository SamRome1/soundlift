"""python -m soundlift <youtube-url>   |   python -m soundlift serve"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Third-party libraries are chatty (pydub regexes, mel filter notes, HF progress bars).
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from . import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="soundlift", description="Pull the music and sound effects out of a YouTube video.")
    ap.add_argument("url", help="YouTube URL, or 'serve' to start the web app")
    ap.add_argument("--out", default="out", help="output folder (default: ./out)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    if a.url == "serve":
        from .server import serve

        serve(Path(a.out), a.port)
        return

    def progress(stage: str, frac: float) -> None:
        print(f"\r{stage:<10} {frac * 100:5.1f}%", end="", file=sys.stderr, flush=True)
        if frac >= 1:
            print(file=sys.stderr)

    res = pipeline.run(a.url, Path(a.out), progress)
    m = res.manifest
    print(f"\n{m['title']} — {m['channel']}")
    for it in m["items"]:
        times = ", ".join(f"{int(s // 60)}:{int(s % 60):02d}" for s, _ in it["ranges"][:6])
        who = f" — {it['artist']}" if it.get("artist") else ""
        print(f"  [{it['kind']:5}] {it['title']}{who}  @ {times}  -> {res.out_dir / it['file']}")
    if m["description_credits"]:
        print("\nCredits found in description:")
        for l in m["description_credits"]:
            print("  " + l)


if __name__ == "__main__":
    main()
