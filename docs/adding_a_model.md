# Adding a model adapter

An adapter is a subclass of `vlaquantbench.models.base.VLAAdapter` registered in
`vlaquantbench/registry.py`. It lives in the model's *own* conda environment
(the core package has no dependency beyond torch/numpy/pyyaml), so import the
official code lazily inside the methods.

```python
from vlaquantbench.components import ComponentMap
from vlaquantbench.models.base import Observation, TaskSpec, VLAAdapter

class MyVLAAdapter(VLAAdapter):
    name = "myvla"
    default_dtype = "bf16"                 # precision of the official release
    supported_benchmarks = ("libero",)

    def load(self):
        from myvla import load_model          # official loader, unchanged hyper-parameters
        self.model = load_model(self.checkpoint).to(self.device, self.dtype).eval()

    def build_component_map(self) -> ComponentMap:
        m = self.model
        cm = ComponentMap()
        cm.add("ve", "vision_tower", m.vision_tower)
        cm.add("mp", "projector", m.projector)
        cm.add("llm", "language_model", m.language_model)   # lm_head is excluded by default pattern
        cm.add("ah", "action_head", m.action_head)
        cm.add("ah", "proprio_proj", m.proprio_proj)          # everything feeding the head
        cm.notes["ah"] = "MLP head + proprio projector"
        return cm

    def reset(self, task: TaskSpec):
        self._queue = []                                      # action chunk queue, history, ...

    def act(self, obs: Observation, task: TaskSpec):
        if not self._queue:
            img = preprocess(obs.images["agentview"])         # the OFFICIAL preprocessing for this benchmark
            chunk = self.model.predict(img, obs.instruction, obs.state)
            self._queue = list(chunk[: self.n_exec])          # re-plan exactly like the official eval code
        return self._queue.pop(0)                             # one env action, benchmark convention
```

Rules:

1. **Component roots must be disjoint** (`ComponentMap.check_disjoint()` is run
   automatically). If a parent module contains two components, list its
   children instead of the parent.
2. Anything with `nn.Linear` layers must belong to some component unless it is
   deliberately left at baseline precision — document such choices in
   `cm.notes`.
3. `act()` returns **one** environment action in the benchmark's convention
   (see the runner docstrings). Chunking, ensembling and un-normalisation
   follow the official evaluation code of the model.
4. Never change model hyper-parameters between settings; quantization is the
   only variable.
5. Add a `configs/models/<name>.yaml` with `env`, `dtype` and the checkpoint
   per benchmark/suite, then `vqb inspect --model <name>` to verify the
   decomposition and parameter counts.
