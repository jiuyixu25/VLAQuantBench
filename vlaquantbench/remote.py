"""Policy server / client so that a model and a simulator can live in different
environments (the X-VLA deployment pattern, and the only option for CALVIN's
Python-3.8 stack).

Server (model environment)::

    vqb serve --model xvla --checkpoint 2toINF/X-VLA-Calvin-ABC_D --preset W4 --scope e2e --port 8010

Client (simulator environment)::

    vqb run --model remote --model-kwargs url=http://127.0.0.1:8010 --benchmark calvin --suite ABC_D --preset W4

Quantization is applied on the server; the client adapter mirrors the served
adapter's protocol attributes (``libero_max_steps`` ...) so runners behave
identically, forwards observations (numpy arrays are base64-encoded ``.npy``
blobs) and records the server's per-inference latency and peak VRAM.
Only the standard library and numpy are needed on the client side.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np

from .components import ComponentMap
from .models.base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)

PROTOCOL_ATTRS = (
    "libero_max_steps", "libero_camera_size", "libero_action_mode",
    "simpler_control_mode", "simpler_max_steps_factor", "calvin_ep_len", "supported_benchmarks", "name",
)


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #
def encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(obj), allow_pickle=False)
        return {"__ndarray__": base64.b64encode(buf.getvalue()).decode("ascii")}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(o) for o in obj]}
    if isinstance(obj, list):
        return [encode(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items() if _serialisable(v)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return None  # dropped (simulator handles, modules, ...)


def _serialisable(v: Any) -> bool:
    return isinstance(v, (np.ndarray, np.generic, tuple, list, dict, str, int, float, bool)) or v is None


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.load(io.BytesIO(base64.b64decode(obj["__ndarray__"])), allow_pickle=False)
        if "__tuple__" in obj:
            return tuple(decode(o) for o in obj["__tuple__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(o) for o in obj]
    return obj


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
class PolicyServer:
    def __init__(self, adapter: VLAAdapter, quant_report: dict[str, Any], *, host: str = "0.0.0.0", port: int = 8010):
        self.adapter = adapter
        self.quant_report = quant_report
        self.host, self.port = host, port
        self.lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        a = self.adapter
        return {
            "describe": a.describe(),
            "quant_report": self.quant_report,
            "protocol": {k: getattr(a, k) for k in PROTOCOL_ATTRS if hasattr(a, k)},
            "dtype": str(a.dtype).replace("torch.", ""),
            "checkpoint": a.checkpoint,
        }

    def handle(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import torch

        from .profiling import peak_memory_gb, reset_peak_memory

        with self.lock:
            if path == "/info":
                return self.info()
            if path == "/reset":
                task = TaskSpec(**payload["task"])
                self.adapter.new_inference_meter(warmup=1)
                reset_peak_memory()
                self.adapter.reset(task)
                return {"ok": True}
            if path == "/act":
                task = TaskSpec(**payload["task"])
                o = payload["obs"]
                obs = Observation(
                    images={k: decode(v) for k, v in o["images"].items()},
                    instruction=o["instruction"],
                    state=decode(o["state"]) if o.get("state") is not None else None,
                    step=int(o.get("step", 0)),
                    raw=decode(o.get("raw")),
                )
                n0 = self.adapter.inference_meter._seen
                t0 = time.perf_counter()
                action = self.adapter.act(obs, task)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                inferred = self.adapter.inference_meter._seen > n0
                return {"action": encode(action), "inference": inferred, "wall_s": wall}
            if path == "/stats":
                return {"peak_vram_gb": peak_memory_gb(), "inference": self.adapter.inference_meter.summary()}
        raise KeyError(path)

    def serve_forever(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quieter
                log.debug(fmt, *args)

            def _send(self, code: int, obj: dict[str, Any]) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    self._send(200, server.handle(self.path, {}))
                except Exception as e:  # pragma: no cover
                    log.exception("GET %s failed", self.path)
                    self._send(500, {"error": repr(e)})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                try:
                    self._send(200, server.handle(self.path, payload))
                except Exception as e:
                    log.exception("POST %s failed", self.path)
                    self._send(500, {"error": repr(e)})

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        log.info("policy server for %s listening on http://%s:%d", self.adapter.name, self.host, self.port)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()


# --------------------------------------------------------------------------- #
# client adapter
# --------------------------------------------------------------------------- #
class RemoteAdapter(VLAAdapter):
    """Client-side stand-in for an adapter served by ``vqb serve``."""

    name: ClassVar[str] = "remote"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero", "simpler", "calvin", "vlabench")

    def __init__(self, checkpoint: str = "remote", *, device="cpu", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device="cpu", dtype="fp32", seed=seed, **kwargs)
        self.url = str(kwargs.get("url", "http://127.0.0.1:8010")).rstrip("/")
        self.timeout = float(kwargs.get("timeout", 600))
        self.info: dict[str, Any] = {}
        self.served_name = "remote"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        if "error" in out:
            raise RuntimeError(f"policy server error on {path}: {out['error']}")
        return out

    def load(self) -> None:
        self.info = self._post("/info", {})
        proto = self.info.get("protocol", {})
        for k, v in proto.items():
            if k in ("name", "supported_benchmarks"):
                continue
            setattr(self, k, v)  # mirror libero_max_steps, calvin_ep_len, ...
        self.served_name = proto.get("name", "remote")
        self.checkpoint = self.info.get("checkpoint", self.checkpoint)
        self.dtype_name = self.info.get("dtype", "")
        self.model = None
        log.info("connected to policy server %s serving %s (%s)", self.url, self.served_name, self.checkpoint)

    def build_component_map(self) -> ComponentMap:
        return ComponentMap(notes={"remote": "quantization is applied on the server; see header.quant_report"})

    def component_map(self) -> ComponentMap:  # no model locally
        if self._cmap is None:
            self._cmap = self.build_component_map()
        return self._cmap

    def describe(self) -> dict[str, Any]:
        return self.info.get("describe", {"model": "remote"})

    def reset(self, task: TaskSpec) -> None:
        self._post("/reset", {"task": asdict(task)})

    def act(self, obs: Observation, task: TaskSpec):
        payload = {
            "task": asdict(task),
            "obs": {
                "images": {k: encode(v) for k, v in obs.images.items()},
                "instruction": obs.instruction,
                "state": encode(np.asarray(obs.state)) if obs.state is not None else None,
                "step": obs.step,
                "raw": encode(obs.raw) if isinstance(obs.raw, dict) else None,
            },
        }
        out = self._post("/act", payload)
        if out.get("inference"):
            m = self.inference_meter
            m._seen += 1
            if m._seen > m.warmup:
                m.samples.append(float(out["wall_s"]))
        return decode(out["action"])

    def peak_vram_gb(self) -> float | None:
        try:
            return self._post("/stats", {}).get("peak_vram_gb")
        except Exception:  # pragma: no cover
            return None
