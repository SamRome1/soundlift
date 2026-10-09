"""YouTube URL -> identified music + extracted sound effects.

Stages:
  1. download   yt-dlp grabs the audio + metadata
  2. separate   MRX (MERL "Cocktail Fork", fast) splits the soundtrack into speech / music / sfx stems
  3. tag        AudioSet classifier (AST) builds a music timeline from the music stem
  4. identify   Shazam fingerprints the music regions
  5. sfx        onsets on the sfx stem, kept if they're loud relative to the full mix,
                then labelled + deduplicated by AST
  6. clean      BandIt Plus re-separates just the clip ranges; averaged with MRX for the cleanest clips
  7. export     clips + manifest.json
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

SR = 44100
YTDLP = [sys.executable, "-m", "yt_dlp"]
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet index ranges (see the model's id2label)
MUSIC_IDX = [i for i in range(137, 283) if not (200 <= i <= 206 or i == 177)]  # bells/gongs count as SFX
VOICE_IDX = list(range(0, 16)) + list(range(27, 38))  # speech + singing: leftover vocal bleed, never SFX
HUMAN_IDX = list(range(0, 51))  # voice + breaths, gasps, coughs: bleed when the speaker is talking
MUSIC = 137
GENERIC_LABELS = {
    "Sound effect", "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", "Echo", "Noise",
    "Environmental noise", "Silence", "Field recording", "Television", "Radio", "Vehicle",
}

# Sounds that are usually just the location (street, wind, crowd), not an effect someone added in the edit.
# Kept, but reported as kind="ambient" so the SFX list stays about designed effects.
AMBIENT_LABELS = {
    "Vehicle", "Car", "Car passing by", "Engine", "Idling", "Accelerating, revving, vroom", "Traffic noise, roadway noise",
    "Train", "Subway, metro, underground", "Bus", "Truck", "Motorcycle", "Emergency vehicle", "Police car (siren)",
    "Ambulance (siren)", "Fire engine, fire truck (siren)", "Aircraft", "Fixed-wing aircraft, airplane", "Helicopter", "Boat, Water vehicle",
    "Wind", "Wind noise (microphone)", "Rustling leaves", "Rain", "Raindrop", "Rain on surface", "Stream", "Ocean",
    "Waves, surf", "Water", "Crowd", "Chatter", "Hubbub, speech noise, speech babble", "Children playing",
    "Walk, footsteps", "Run", "Shuffle", "Door", "Sliding door", "Cupboard open or close", "Drawer open or close",
    "Dishes, pots, and pans", "Cutlery, silverware", "Bird", "Bird vocalization, bird call, bird song", "Pigeon, dove",
    "Crow", "Caw", "Animal", "Dog", "Bark", "Domestic animals, pets", "Air conditioning", "Mechanical fan",
    "Typing", "Computer keyboard", "Rub", "Scrape", "Wind chime",
}
MUSIC_LIBRARIES = re.compile(
    r"epidemic|artlist|soundstripe|musicbed|uppbeat|audiio|pixabay|bensound|ncs|nocopyright|"
    r"envato|audiojungle|storyblocks|freesound|zapsplat|soundsnap|spotify|music\.apple|soundcloud|bandcamp",
    re.I,
)
# "Music: ...", "Song - ...", "♪ ..." style credit lines (not SEO keyword spam like "sound effects for edits")
CREDIT_LINE = re.compile(r"^\W*(music|songs?|tracks?|bgm|beats?|sfx|sound ?effects?)( by| used| credits?| links?)?\s*[:\-–—|]|[♪🎵🎶🎧]", re.I)
URL = re.compile(r"https?://\S+")

Progress = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- download


def video_info(url: str) -> dict:
    out = subprocess.run([*YTDLP, "-J", "--no-playlist", "--no-warnings", url], capture_output=True, text=True)
    if out.returncode != 0:
        msg = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "yt-dlp failed"
        raise RuntimeError(f"Couldn't read that link: {msg}")
    return json.loads(out.stdout)


def download_audio(url: str, work: Path, attempts: int = 4) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    cmd = [*YTDLP, "--no-playlist", "--no-warnings", "-q", "--retries", "5",
           "-f", "ba[format_note*=original][protocol^=http]/ba[format_note!*=dubbed][protocol^=http]/ba/b",  # skip AI dubs
           "-o", str(work / "source.%(ext)s"), url]
    for i in range(attempts):  # YouTube hands out the odd 403; a fresh request usually works
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            break
        if i == attempts - 1:
            raise RuntimeError(f"Couldn't download the video: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'yt-dlp failed'}")
        for p in work.glob("source.*"):
            p.unlink()
        time.sleep(2 * (i + 1))
    src = next(p for p in work.iterdir() if p.name.startswith("source."))
    wav = work / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "2", "-ar", str(SR), str(wav)],
        check=True,
    )
    src.unlink()
    return wav


def description_credits(description: str) -> list[str]:
    """Lines in the description that look like music/SFX credits."""
    lines = [l.strip() for l in (description or "").splitlines()]
    hits = []
    for l in lines:
        url = URL.search(l)
        if CREDIT_LINE.search(l) or (url and MUSIC_LIBRARIES.search(url.group())):
            hits.append(l)
    return hits[:30]


# ---------------------------------------------------------------- separate

MRX_URL = (
    "https://media.githubusercontent.com/media/merlresearch/cocktail-fork-separation/main/"
    "checkpoints/adapted_loudness_mrx_pre_trained_weights.pth"
)
CACHE = Path(os.environ.get("SOUNDLIFT_CACHE", Path.home() / ".cache" / "soundlift"))


def _mrx_weights() -> Path:
    return _cached(MRX_URL, "mrx_adapted_loudness.pth")


BANDIT_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/"
    "model_bandit_plus_dnr_sdr_11.47.chpt"
)
BANDIT_CONFIG = dict(
    in_channel=1, stems=["speech", "music", "effects"], band_specs="musical", n_bands=64, fs=44100,
    require_no_overlap=False, require_no_gap=True, normalize_channel_independently=False,
    treat_channel_as_feature=True, n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True,
    rnn_type="GRU", mlp_dim=512, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True,
    n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window", wkwargs=None, power=None,
    center=True, normalized=True, pad_mode="constant", onesided=True,
)


def _cached(url: str, name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        import urllib.request

        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def separate_bandit(audio: np.ndarray, overlap: int = 2, batch: int = 4) -> dict[str, np.ndarray]:
    """[n, 2] mix -> {"speech", "music", "sfx"}: [n, 2]. BandIt Plus (DnR SDR 11.5), 6s windows overlap-added."""
    import torch

    from .vendor.bandit import MultiMaskMultiSourceBandSplitRNNSimple

    dev = _device()
    model = MultiMaskMultiSourceBandSplitRNNSimple(**BANDIT_CONFIG).eval()
    model.load_state_dict(torch.load(_cached(BANDIT_URL, "bandit_plus_dnr.chpt"), map_location="cpu"))
    model = model.float().to(dev)  # some filterbank buffers are float64, which MPS can't hold

    c = 264600  # the 6s window the model was trained on
    h = c // overlap
    n = len(audio)
    pad = c  # pad both ends so every sample gets full window coverage
    x = np.pad(audio.T.astype(np.float32), ((0, 0), (pad, pad + (-(n + 2 * pad - c) % h))))
    starts = list(range(0, x.shape[1] - c + 1, h))
    win = torch.hann_window(c, periodic=False) + 1e-3
    out = torch.zeros(3, 2, x.shape[1])
    wsum = torch.zeros(x.shape[1])
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            idx = starts[i : i + batch]
            y = model(torch.stack([xt[:, s : s + c] for s in idx]).to(dev)).cpu()  # [b, 3, 2, c]
            for s, yy in zip(idx, y):
                out[..., s : s + c] += yy * win
                wsum[s : s + c] += win
    out = (out / wsum)[..., pad : pad + n]
    return {name: out[i].numpy().T for i, name in enumerate(["speech", "music", "sfx"])}


def separate(audio: np.ndarray, chunk: float = 30.0, overlap: float = 1.0) -> dict[str, np.ndarray]:
    """[n, 2] mix -> {"music", "speech", "sfx"}: [n, 2] each. Chunked with crossfades to bound memory."""
    import pyloudnorm
    import torch

    from .vendor.mrx import MRX

    dev = _device()
    model = MRX().eval()
    model.load_state_dict(torch.load(_mrx_weights(), map_location="cpu"))
    model = model.to(dev)

    # The model expects soundtrack-level loudness (~-27 LUFS); undo the gain afterwards.
    loud = pyloudnorm.Meter(SR).integrated_loudness(audio)
    gain = float(10 ** ((-27 - loud) / 20)) if np.isfinite(loud) else 1.0
    x = torch.from_numpy((audio * gain).T.astype(np.float32))

    n, c, o = x.shape[1], int(chunk * SR), int(overlap * SR)
    out = torch.zeros(3, 2, n)
    weight = torch.zeros(n)
    fade = torch.linspace(0, 1, o)
    with torch.no_grad():
        for start in range(0, n, c - o):
            end = min(n, start + c)
            y = model(x[:, start:end].to(dev)).cpu()
            w = torch.ones(end - start)
            if start > 0:
                w[: min(o, len(w))] = fade[: min(o, len(w))]
            if end < n:
                w[-o:] = torch.minimum(w[-o:], fade.flip(0))
            out[..., start:end] += y * w
            weight[start:end] += w
            if end == n:
                break
    out = out / weight.clamp(min=1e-6) / gain
    return {name: out[i].numpy().T for i, name in enumerate(["music", "speech", "sfx"])}


# ---------------------------------------------------------------- tagging


class Tagger:
    """AudioSet classifier. Gives per-clip class probabilities + an embedding for dedup."""

    def __init__(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.device = _device()
        self.fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
        self.model = ASTForAudioClassification.from_pretrained(AST_MODEL).to(self.device).eval()
        self.labels = self.model.config.id2label
        self.label_idx = {v: k for k, v in self.labels.items()}

    def run(self, clips: list[np.ndarray], batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """clips: mono float32 @ 44.1k. Returns (probs [N,527], embeddings [N,768])."""
        probs, embs = [], []
        for i in range(0, len(clips), batch):
            chunk = [resample_poly(c, 160, 441).astype(np.float32) for c in clips[i : i + batch]]
            inputs = self.fe(chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            probs.append(self.torch.sigmoid(out.logits).cpu().numpy())
            embs.append(out.hidden_states[-1][:, :2].mean(1).cpu().numpy())
        if not probs:
            return np.zeros((0, 527)), np.zeros((0, 768))
        return np.concatenate(probs), np.concatenate(embs)

    def sfx_label(self, p: np.ndarray) -> tuple[str, float]:
        """Best non-music, non-voice label; prefers a specific one ("Whoosh") over a generic one."""
        q = p.copy()
        q[MUSIC_IDX] = 0
        q[VOICE_IDX] = 0
        i = int(q.argmax())
        g = q.copy()
        g[[self.label_idx[l] for l in GENERIC_LABELS if l in self.label_idx]] = 0
        j = int(g.argmax())
        if g[j] >= 0.03 and g[j] >= q[i] * 0.3:
            i = j
        return self.labels[i], float(q[i])


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9))


def music_timeline(music: np.ndarray, tagger: Tagger, win: float = 3.0, hop: float = 1.5):
    """Returns list of (start, end, music_prob) windows."""
    n, w, h = len(music), int(win * SR), int(hop * SR)
    starts = list(range(0, max(1, n - w + 1), h))
    clips = [music[s : s + w] for s in starts]
    probs, _ = tagger.run(clips)
    out = []
    for s, c, p in zip(starts, clips, probs):
        mp = float(p[MUSIC]) if _db(c) > -50 else 0.0
        out.append((s / SR, (s + w) / SR, mp))
    return out


def music_regions(timeline, thresh: float = 0.35, min_len: float = 3.0, max_gap: float = 3.0):
    regions: list[list[float]] = []
    for s, e, p in timeline:
        if p < thresh:
            continue
        if regions and s - regions[-1][1] <= max_gap:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    return [(s, e) for s, e in regions if e - s >= min_len]


# ---------------------------------------------------------------- identify


def _wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, resample_poly(x, 160, 441).astype(np.float32), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def _shazam_windows(windows: list[tuple[float, np.ndarray, np.ndarray]], progress: Progress):
    from shazamio import Shazam

    sh = Shazam()
    results = []
    for i, (t, music_clip, mix_clip) in enumerate(windows):
        track = None
        for clip in (music_clip, mix_clip):  # isolated music first, full mix as fallback
            for attempt in range(3):
                try:
                    res = await sh.recognize(_wav_bytes(clip))
                    track = res.get("track")
                    break
                except Exception:
                    await asyncio.sleep(2 * (attempt + 1))
            if track:
                break
            await asyncio.sleep(0.6)
        results.append((t, track))
        progress("identify", (i + 1) / len(windows))
    return results


def identify(music, mix, regions, progress: Progress = _noop, win: float = 12.0, step: float = 10.0):
    """Shazam each music region in windows. Returns (tracks, unidentified_ranges)."""
    windows, owner = [], []
    for ri, (s, e) in enumerate(regions):
        t = s
        while True:
            a, b = int(t * SR), int(min(t + win, e) * SR)
            windows.append((t, music[a:b], mix[a:b]))
            owner.append(ri)
            t += step
            if t + 4 > e:
                break
    hits = asyncio.run(_shazam_windows(windows, progress)) if windows else []

    # A real song matches window after window. A lone hit in a region of misses is
    # almost always a false positive, so require 2+ hits unless the region is tiny.
    per_region: dict[int, int] = {}
    for ri in owner:
        per_region[ri] = per_region.get(ri, 0) + 1
    counts: dict[tuple[int, str], int] = {}
    for (t, track), ri in zip(hits, owner):
        if track:
            counts[(ri, track["key"])] = counts.get((ri, track["key"]), 0) + 1

    def accepted(ri: int, key: str) -> bool:
        c = counts[(ri, key)]
        return c >= 2 or c / per_region[ri] >= 0.5

    tracks: dict[str, dict] = {}
    unmatched = []
    for (t, track), ri in zip(hits, owner):
        s, e = regions[ri]
        span = [t, min(t + step, e)] if t + step + 4 <= e else [t, e]
        if not track or not accepted(ri, track["key"]):
            unmatched.append(span)
            continue
        tr = tracks.setdefault(track["key"], {
            "kind": "music",
            "title": track.get("title"),
            "artist": track.get("subtitle"),
            "shazam_url": track.get("url"),
            "cover": (track.get("images") or {}).get("coverart"),
            "apple_music_url": _apple_url(track),
            "ranges": [],
        })
        tr["ranges"].append(span)

    for tr in tracks.values():
        tr["ranges"] = _merge(tr["ranges"], gap=1)
        q = f"{tr['title']} {tr['artist']}"
        tr["search"] = {
            "spotify": f"https://open.spotify.com/search/{_q(q)}",
            "youtube": f"https://www.youtube.com/results?search_query={_q(q)}",
        }
    unidentified = [r for r in _merge(unmatched, gap=12) if r[1] - r[0] >= 5]  # short gaps = same bed
    return list(tracks.values()), unidentified


def _apple_url(track: dict) -> str | None:
    for opt in (track.get("hub") or {}).get("options", []) or []:
        for a in opt.get("actions", []) or []:
            if "music.apple.com" in (a.get("uri") or ""):
                return a["uri"]
    return None


def _q(s: str) -> str:
    from urllib.parse import quote

    return quote(s)


def _merge(ranges, gap: float):
    ranges = sorted(ranges)
    out: list[list[float]] = []
    for s, e in ranges:
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ---------------------------------------------------------------- sfx


def _env_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)


def sfx_candidates(fx: np.ndarray, mix: np.ndarray, hop: int = 441, min_rel: float = -6.0):
    """Onsets in the sfx stem. Returns [(start, end, rel_db)].

    rel_db = how loud the sfx stem is vs. the full mix at that moment. Separation leaks a bit of
    speech/music into the sfx stem, but leakage sits 10-20 dB under the mix; real effects don't.
    """
    db = _env_db(fx, hop)
    rel = median_filter(db - _env_db(mix, hop), size=5)
    base = median_filter(db, size=301, mode="nearest")  # 3s rolling floor
    fps = SR / hop
    n = len(db)

    out, i = [], 5
    while i < n:
        if db[i] > -50 and db[i] - base[i] > 8 and db[i] - db[i - 5] > 6:
            start, peak, j = i - 3, i, i
            limit = min(n, i + int(5 * fps))
            while j < limit:
                if db[j] > db[peak]:
                    peak = j
                if j > peak + 3 and db[j] < max(base[j] + 3, db[peak] - 35):
                    break
                j += 1
            r = float(rel[start:j].max())
            if (j - start) / fps >= 0.08 and r >= min_rel:
                out.append((max(0, start) / fps, min(j + 10, n) / fps, r))
            i = j + 5
        else:
            i += 1
    return out


def detect_sfx(fx: np.ndarray, mix: np.ndarray, speech: np.ndarray, tagger: Tagger, progress: Progress = _noop):
    cands = sfx_candidates(fx, mix)
    progress("sfx", 0.2)
    clips = [fx[int(s * SR) : int(e * SR)] for s, e, _ in cands]
    probs, embs = tagger.run(clips)
    progress("sfx", 0.8)

    events = []
    for (s, e, rel), p, emb in zip(cands, probs, embs):
        label, lp = tagger.sfx_label(p)
        voice, music = float(p[VOICE_IDX].max()), float(p[MUSIC_IDX].max())
        if voice > 2 * lp or (music > 3 * lp and rel < 0):
            continue  # leaked speech, or a musical hit that slipped into the sfx stem
        a, b = int(s * SR), int(e * SR)
        talking = _db(speech[a:b]) - _db(mix[a:b]) > -10
        if talking and int(p.argmax()) in HUMAN_IDX or (talking and tagger.label_idx[label] in HUMAN_IDX):
            continue  # breaths/gasps from the speaker
        events.append({"start": s, "end": e, "label": label, "conf": lp, "rel": rel,
                       "emb": emb / (np.linalg.norm(emb) + 1e-9)})

    # Same effect reused across the video -> one entry with many timestamps.
    groups: list[dict] = []
    for ev in events:
        for g in groups:
            if float(g["emb"] @ ev["emb"]) > 0.9:
                g["hits"].append(ev)
                break
        else:
            groups.append({"emb": ev["emb"], "hits": [ev]})

    out = []
    for g in groups:
        best = max(g["hits"], key=lambda h: (h["rel"], h["conf"]))  # cleanest instance
        out.append({
            "kind": "ambient" if best["label"] in AMBIENT_LABELS else "sfx",
            "title": best["label"],
            "confidence": round(best["conf"], 3),
            "clip_range": [best["start"], best["end"]],
            "ranges": [[round(h["start"], 2), round(h["end"], 2)] for h in g["hits"]],
        })
    out.sort(key=lambda x: x["ranges"][0][0])
    progress("sfx", 1.0)
    return out


# ---------------------------------------------------------------- export


def write_clip(x: np.ndarray, path: Path, fade: float = 0.01) -> None:
    x = x.copy()
    f = min(int(fade * SR), len(x) // 2)
    if f:
        ramp = np.linspace(0, 1, f)[:, None] if x.ndim == 2 else np.linspace(0, 1, f)
        x[:f] *= ramp
        x[-f:] *= ramp[::-1]
    peak = np.abs(x).max()
    if peak > 0:
        x *= 0.89 / peak  # -1 dBFS
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16")


def gate_tail(x: np.ndarray, floor_db: float = -40.0, hop: int = 441) -> np.ndarray:
    """Fade out whatever sits far below the effect's peak (separation bleed), and trim the silence."""
    m = x.mean(1) if x.ndim == 2 else x
    n = len(m) // hop
    if n < 3:
        return x
    db = 20 * np.log10(np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    loud = np.where(db > db.max() + floor_db)[0]
    a, b = loud[0], loud[-1] + 1
    gain = np.clip((db - (db.max() + floor_db)) / 12, 0, 1)  # 12 dB soft knee
    gain = np.convolve(gain, np.ones(5) / 5, mode="same")  # smooth so it doesn't chatter
    g = np.repeat(gain, hop)
    g = np.concatenate([g, np.full(len(m) - len(g), g[-1])])
    y = x * (g[:, None] if x.ndim == 2 else g)
    return y[max(0, a - 2) * hop : min(n, b + 5) * hop]


def quietest_window(ranges, noise: np.ndarray, length: float = 30.0, hop: float = 1.0) -> tuple[float, float]:
    """Inside the given ranges, the window where `noise` (speech + sfx energy) is lowest."""
    best, best_score = None, np.inf
    for s, e in ranges:
        w = min(length, e - s)
        t = s
        while t + w <= e + 1e-6:
            seg = noise[int(t * SR) : int((t + w) * SR)]
            score = float(np.mean(seg**2)) if len(seg) else np.inf
            if score < best_score - 1e-12 or (best and w > best[1] - best[0] and score <= best_score * 1.5):
                best, best_score = (t, t + w), score
            t += hop
    return best or ranges[0]


def clean_windows(mix: np.ndarray, mrx: dict[str, np.ndarray], windows: list[tuple[float, float]],
                  context: float = 3.0, progress: Progress = _noop) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Re-separate just these time windows with BandIt and average it with the MRX result.

    The two models fail on different sounds (BandIt drops some booms/whooshes into the music stem,
    MRX leaks more on clicks/risers), so the average beats either alone: on a synthetic benchmark,
    SFX clips went from 3.5 dB (MRX) / -0.6 dB (BandIt) to 7.3 dB SI-SDR, music clips 3.3 -> 5.6 dB.

    Overlapping windows are merged and each gets a few seconds of real context on both sides.
    Returns [(start_sample, stems)] spans; use `cut` to pull a clip out of them.
    """
    spans = _merge([[max(0.0, a - context), min(len(mix) / SR, b + context)] for a, b in windows], gap=context)
    total = sum(e - s for s, e in spans) or 1.0
    out, done = [], 0.0
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        bandit = separate_bandit(mix[a:b])
        out.append((a, {k: 0.5 * (_level_match(mrx[k][a:b], bandit[k]) + bandit[k]) for k in bandit}))
        done += e - s
        progress("clean", done / total)
    return out


def _level_match(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale x to ref's level (the adapted MRX checkpoint doesn't preserve output level)."""
    d = float(x.ravel() @ x.ravel())
    return x * max(0.0, float(x.ravel() @ ref.ravel()) / d) if d > 0 else x


def cut(spans, stem: str, s: float, e: float) -> np.ndarray:
    a, b = int(s * SR), int(e * SR)
    for start, stems in spans:
        x = stems[stem]
        if start <= a and b <= start + len(x):
            return x[a - start : b - start]
    raise ValueError(f"{s:.2f}-{e:.2f}s was not cleaned")


@dataclass
class Result:
    out_dir: Path
    manifest: dict = field(default_factory=dict)


def run(url: str, out_root: Path, progress: Progress = _noop) -> Result:
    t0 = time.time()
    progress("download", 0.0)
    info = video_info(url)
    vid = info["id"]
    out = out_root / vid
    if (out / "manifest.json").exists():
        return Result(out, json.loads((out / "manifest.json").read_text()))
    if out.exists():
        shutil.rmtree(out)
    work = out / "_work"

    audio_path = download_audio(url, work)
    mix, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    progress("download", 1.0)

    progress("separate", 0.0)
    stems = separate(mix)
    progress("separate", 1.0)
    mono = {k: v.mean(1) for k, v in stems.items()}
    mix_m = mix.mean(1)

    progress("tag", 0.0)
    tagger = Tagger()
    regions = music_regions(music_timeline(mono["music"], tagger))
    progress("tag", 1.0)

    progress("identify", 0.0)
    tracks, unidentified = identify(mono["music"], mix_m, regions, progress)
    progress("identify", 1.0)

    progress("sfx", 0.0)
    sfx = detect_sfx(mono["sfx"], mix_m, mono["speech"], tagger, progress)

    # Pick every clip's time range first, then re-separate only those ranges with the better model.
    noise = mono["speech"] + mono["sfx"]
    for tr in tracks:
        tr["clip_range"] = list(quietest_window(tr["ranges"], noise))
    unknown = [{"kind": "music", "title": f"Unidentified music {i}", "artist": None, "identified": False,
                "ranges": [[s, e]], "clip_range": list(quietest_window([(s, e)], noise))}
               for i, (s, e) in enumerate(unidentified, 1)]
    progress("clean", 0.0)
    spans = clean_windows(mix, stems, [tuple(x["clip_range"]) for x in tracks + unknown + sfx], progress=progress)

    progress("export", 0.0)
    items = []
    for i, tr in enumerate(tracks, 1):
        s, e = tr.pop("clip_range")
        tr["identified"] = True
        tr["file"] = f"clips/music_{i:02d}_{_slug(tr['title'] or 'track')}.wav"
        write_clip(cut(spans, "music", s, e), out / tr["file"], fade=0.3)
        items.append(tr)
    for i, tr in enumerate(unknown, 1):
        s, e = tr.pop("clip_range")
        tr["file"] = f"clips/music_unknown_{i:02d}.wav"
        write_clip(cut(spans, "music", s, e), out / tr["file"], fade=0.3)
        items.append(tr)
    for i, fx in enumerate(sfx, 1):
        s, e = fx.pop("clip_range")
        fx["file"] = f"clips/{fx['kind']}_{i:02d}_{_slug(fx['title'])}.wav"
        write_clip(gate_tail(cut(spans, "sfx", s, e)), out / fx["file"])
        items.append(fx)

    for name in ("music", "sfx", "speech"):
        write_clip(stems[name], out / "stems" / f"{name}.wav", fade=0)

    manifest = {
        "id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "description_credits": description_credits(info.get("description", "")),
        "items": items,
        "stems": {k: f"stems/{k}.wav" for k in ("music", "sfx", "speech")},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    progress("export", 1.0)
    return Result(out, manifest)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:30]
