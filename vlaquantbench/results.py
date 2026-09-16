"""Result records and JSONL persistence.

One run = one JSONL file. The first line is a :class:`RunHeader`; every
following line is an :class:`EpisodeRecord`. Keeping per-episode outcomes
(instead of only aggregate success rates) is what makes confidence intervals
and resumable runs possible.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = ["RunHeader", "EpisodeRecord", "RunWriter", "read_run", "iter_runs", "git_commit", "gpu_name"]

log = logging.getLogger(__name__)


def git_commit(path: str | os.PathLike | None = None) -> str | None:
    try:
        root = Path(path or Path(__file__).resolve().parents[1])
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return None


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return None


@dataclass
class RunHeader:
    kind: str = "header"
    model: str = ""
    checkpoint: str = ""
    benchmark: str = ""
    suite: str = ""
    preset: str = "BASELINE"
    scope: str = "e2e"
    method: str = "rtn"
    baseline_dtype: str = ""
    episodes_per_task: int = 0
    seed: int = 0
    quantize_lm_head: bool = False
    include_conv: bool = False
    quant_report: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    hostname: str = field(default_factory=socket.gethostname)
    gpu: str | None = field(default_factory=gpu_name)
    python: str = field(default_factory=platform.python_version)
    vqb_commit: str | None = field(default_factory=git_commit)

    def run_id(self) -> str:
        return f"{self.benchmark}-{self.suite or 'all'}__{self.model}__{self.method}-{self.preset}-{self.scope}"


@dataclass
class EpisodeRecord:
    kind: str = "episode"
    suite: str = ""
    task_id: int = -1
    task_name: str = ""
    episode: int = -1
    seed: int = -1
    success: bool = False
    steps: int = 0
    wall_time_s: float = 0.0
    # benchmark-specific scalar metrics
    subtasks_completed: int | None = None  # CALVIN (0-5)
    progress_score: float | None = None  # VLABench
    # timing around adapter.act() per env step (includes cheap queue pops), after warmup
    policy_latency_mean_s: float | None = None
    policy_latency_p95_s: float | None = None
    # timing of model inference calls only (observation -> action chunk), after warmup
    inference_latency_mean_s: float | None = None
    inference_latency_p95_s: float | None = None
    n_inferences: int | None = None
    peak_vram_gb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.suite, self.task_id, self.episode)


class RunWriter:
    """Append-only JSONL writer with resume support."""

    def __init__(self, path: str | os.PathLike, header: RunHeader, *, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        self.done: set[tuple[str, int, int]] = set()
        if resume and self.path.exists():
            old_header, episodes = read_run(self.path)
            if old_header is not None and not _compatible(old_header, header):
                raise RuntimeError(
                    f"{self.path} holds a run with a different configuration "
                    f"({old_header.run_id()} vs {header.run_id()}); refusing to append"
                )
            self.done = {e.key for e in episodes}
            self._fh = self.path.open("a")
        else:
            self._fh = self.path.open("w")
            self._write(asdict(header))

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, default=_json_default) + "\n")
        self._fh.flush()

    def is_done(self, suite: str, task_id: int, episode: int) -> bool:
        return (suite, task_id, episode) in self.done

    def add(self, rec: EpisodeRecord) -> None:
        self._write(asdict(rec))
        self.done.add(rec.key)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _compatible(a: RunHeader, b: RunHeader) -> bool:
    keys = ("model", "checkpoint", "benchmark", "suite", "preset", "scope", "method", "quantize_lm_head",
            "include_conv", "seed")
    return all(getattr(a, k) == getattr(b, k) for k in keys)


def _json_default(o: Any):
    if dataclasses.is_dataclass(o):
        return asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _from_dict(cls, d: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    known = {k: v for k, v in d.items() if k in names}
    unknown = {k: v for k, v in d.items() if k not in names}
    obj = cls(**known)
    if unknown:
        obj.extra.update(unknown)
    return obj


def read_run(path: str | os.PathLike) -> tuple[RunHeader | None, list[EpisodeRecord]]:
    header: RunHeader | None = None
    episodes: list[EpisodeRecord] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # An unclean shutdown (power loss, reboot mid-write) can leave a
                # truncated or zero-filled trailing line. Skip it so the run
                # stays resumable; the episode it belonged to is simply re-run.
                log.warning("%s: skipping undecodable line (%d chars)", path, len(line))
                continue
            if obj.get("kind") == "header":
                header = _from_dict(RunHeader, obj)
            else:
                episodes.append(_from_dict(EpisodeRecord, obj))
    return header, episodes


def iter_runs(root: str | os.PathLike) -> Iterator[tuple[Path, RunHeader, list[EpisodeRecord]]]:
    for p in sorted(Path(root).rglob("*.jsonl")):
        header, eps = read_run(p)
        if header is not None:
            yield p, header, eps
