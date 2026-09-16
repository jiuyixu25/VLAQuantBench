"""Calibration text for the LLM PTQ methods.

Both AWQ and SmoothQuant calibrate on the Pile validation split
(``mit-han-lab/pile-val-backup``): AWQ uses 128 samples of 512 tokens,
SmoothQuant 512 samples of 512 tokens. The official implementations read
the data themselves; this module only resolves / downloads the file so that
both see the identical default corpus, and offers the same corpus as a list of
strings for code paths that accept raw text.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

PILE_VAL_REPO = "mit-han-lab/pile-val-backup"
PILE_VAL_FILE = "val.jsonl.zst"
DEFAULT_CACHE = Path(os.environ.get("VQB_CALIB_DIR", Path.home() / ".cache" / "vlaquantbench"))


def pile_val_path(path: str | None = None) -> str:
    """Return a local path to ``val.jsonl.zst``; downloads it from the HF hub if needed."""
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return path
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    local = DEFAULT_CACHE / PILE_VAL_FILE
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download  # type: ignore

    log.info("downloading %s/%s to %s", PILE_VAL_REPO, PILE_VAL_FILE, DEFAULT_CACHE)
    got = hf_hub_download(PILE_VAL_REPO, PILE_VAL_FILE, repo_type="dataset", local_dir=str(DEFAULT_CACHE))
    return got


def pile_val_texts(n_samples: int, path: str | None = None, seed: int = 42, min_chars: int = 1000) -> list[str]:
    """``n_samples`` shuffled documents from the Pile validation split."""
    import io
    import json
    import random

    import zstandard as zstd  # type: ignore

    texts: list[str] = []
    with open(pile_val_path(path), "rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                try:
                    t = json.loads(line)["text"]
                except Exception:  # pragma: no cover
                    continue
                if len(t) >= min_chars:
                    texts.append(t)
                if len(texts) >= 20 * n_samples:
                    break
    random.Random(seed).shuffle(texts)
    return texts[:n_samples]
