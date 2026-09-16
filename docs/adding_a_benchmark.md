# Adding a benchmark runner

A runner subclasses `vlaquantbench.benchmarks.base.BenchmarkRunner` and is
registered in `vlaquantbench/registry.py`. It must reproduce the benchmark's
**official** closed-loop protocol: task list, initial states, episode counts,
step limits and success criterion. The shared loop in `BenchmarkRunner.run`
handles bookkeeping, latency measurement and resumable JSONL persistence.

```python
class MyRunner(BenchmarkRunner):
    name = "mybench"
    primary_metric = "success_rate"

    @classmethod
    def suites(cls): return ("default",)
    @classmethod
    def default_episodes_per_task(cls): return 50

    def tasks(self) -> list[TaskSpec]:
        ...  # one TaskSpec per task, with the official max_steps

    def run_episode(self, adapter, task, episode, meter) -> EpisodeOutcome:
        env = self._env(task); obs = env.reset(seed=self.episode_seed(task, episode))
        adapter.reset(task)
        for step in range(task.max_steps):
            o = self.make_obs({"cam": obs["rgb"]}, task.instruction, obs["state"], step, obs)
            with meter.measure():
                action = adapter.act(o, task)
            obs, done, info = env.step(action)
            if done: return EpisodeOutcome(success=True, steps=step + 1)
        return EpisodeOutcome(success=False, steps=task.max_steps)
```

Conventions:

* Pass simulator images **unmodified** except for benchmark-wide conventions
  that every official evaluation applies (LIBERO's 180° rotation). Model
  specific preprocessing belongs to the adapter.
* Put benchmark-specific scalar metrics in `EpisodeOutcome`
  (`subtasks_completed` for CALVIN-style chains, `progress_score` for
  VLABench) and register the metric in `vlaquantbench/summarize.py::METRIC`.
* Seeds: derive everything from `self.seed` and `self.episode_seed(...)` so a
  cell can be reproduced from its JSONL header.
