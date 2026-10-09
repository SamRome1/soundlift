"""Fast tests for the pure-Python parts (no models, no network)."""

import numpy as np

from soundlift import pipeline as P

SR = P.SR


def test_description_credits_keeps_credits_and_drops_seo_spam():
    desc = """Woosh sound effect,
sound effects for edits,
Music: Lakey Inspired - Chill Day
Sound Effect link :- https://drive.google.com/x
🎵 Track: Blue Wednesday - Murmuration
Get 2 months free https://www.epidemicsound.com/referral/abc
Check my gear https://amazon.com/x"""
    assert P.description_credits(desc) == [
        "Music: Lakey Inspired - Chill Day",
        "Sound Effect link :- https://drive.google.com/x",
        "🎵 Track: Blue Wednesday - Murmuration",
        "Get 2 months free https://www.epidemicsound.com/referral/abc",
    ]


def test_merge_joins_close_ranges():
    assert P._merge([[10, 12], [0, 5], [5.5, 8]], gap=1) == [[0, 8], [10, 12]]


def test_music_regions_merges_gaps_and_drops_short_blips():
    timeline = [(0, 3, 0.9), (1.5, 4.5, 0.9), (6, 9, 0.8), (20, 21, 0.9), (30, 33, 0.1)]
    assert P.music_regions(timeline) == [(0, 9)]


def test_quietest_window_avoids_talking():
    noise = np.zeros(100 * SR, np.float32)
    noise[: 40 * SR] = 1.0  # someone talks for the first 40s
    assert P.quietest_window([(0, 100)], noise) == (40.0, 70.0)


def test_gate_tail_trims_bleed_around_an_effect():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2 * SR, 2)).astype(np.float32) * 0.001  # faint bleed
    x[SR // 2 : SR // 2 + 8000] += rng.standard_normal((8000, 2)).astype(np.float32) * 0.5  # the effect
    y = P.gate_tail(x)
    assert len(y) < len(x) / 4
    assert np.abs(y[-2000:]).max() < 0.01


def test_sfx_candidates_finds_a_hit_over_silence():
    rng = np.random.default_rng(0)
    fx = np.zeros(10 * SR, np.float32)
    fx[4 * SR : 4 * SR + 4410] = rng.standard_normal(4410) * 0.5
    found = P.sfx_candidates(fx, fx)
    assert len(found) == 1
    start, end, rel = found[0]
    assert 3.9 < start < 4.05 and end > 4.1


def test_cut_reads_from_the_right_span():
    spans = [(SR * 10, {"sfx": np.arange(SR * 5, dtype=np.float32)})]
    clip = P.cut(spans, "sfx", 11.0, 12.0)
    assert len(clip) == SR and clip[0] == SR
