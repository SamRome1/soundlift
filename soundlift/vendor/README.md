`mrx.py` is copied unmodified from [merlresearch/cocktail-fork-separation](https://github.com/merlresearch/cocktail-fork-separation)
(Copyright (C) 2023 Mitsubishi Electric Research Laboratories, MIT license; see the SPDX header in the file).

The pretrained weights (`adapted_loudness_mrx_pre_trained_weights.pth`) are downloaded from that repo on first run
and cached in `~/.cache/soundlift`.

`bandit/` is the BandIt model code from [kwatcharasupat/bandit](https://github.com/kwatcharasupat/bandit) (Apache 2.0, LICENSE included),
as packaged in [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT).
Changes: imports made package-relative, and the training-only `pytorch_lightning` / `spafe` imports made optional.
The BandIt Plus DnR checkpoint is downloaded from MSST's v.1.0.3 release on first run.
