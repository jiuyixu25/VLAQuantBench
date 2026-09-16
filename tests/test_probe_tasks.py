import pytest
from vlaquantbench.benchmarks.base import TaskSpec
from vlaquantbench.search.probe import select_probe_tasks


class FakeRunner:
    suite = "fake"

    def __init__(self, spec):  # spec: list of (family, n_episodes)
        self._tasks = [TaskSpec(benchmark="f", suite="fake", task_id=i, task_name=f"t{i}",
                                instruction="", max_steps=10, extra={"family": fam, "n_episodes": n})
                       for i, (fam, n) in enumerate(spec)]

    def tasks(self):
        return list(self._tasks)

    def episodes_for(self, task):
        return list(range(int(task.extra["n_episodes"])))


SPEC = [("a", 60)] * 10 + [("b", 25)] * 10 + [("c", 9)] * 10 + [("d", 1)] * 10


def test_filters_tasks_without_enough_episodes():
    r = FakeRunner(SPEC)
    picked = select_probe_tasks(r, 100, blocks=3, episodes_from=0)
    assert all(int(t.extra["n_episodes"]) >= 3 for t in picked)
    assert len(picked) == 30  # the 1-episode family is unusable for 3 blocks


def test_stratifies_across_families_and_is_deterministic():
    r = FakeRunner(SPEC)
    picked = select_probe_tasks(r, 6, blocks=3, episodes_from=0, seed=0)
    assert len(picked) == 6
    fams = {t.extra["family"] for t in picked}
    assert fams == {"a", "b", "c"}                      # every usable family represented
    again = select_probe_tasks(r, 6, blocks=3, episodes_from=0, seed=0)
    assert [t.task_id for t in picked] == [t.task_id for t in again]
    assert picked == sorted(picked, key=lambda t: t.task_id)


def test_respects_episodes_from_offset():
    r = FakeRunner(SPEC)
    picked = select_probe_tasks(r, 100, blocks=3, episodes_from=20)
    assert all(int(t.extra["n_episodes"]) >= 23 for t in picked)   # the 60- and 25-episode families
    assert len(picked) == 20
    with pytest.raises(ValueError, match="no task"):
        select_probe_tasks(r, 5, blocks=3, episodes_from=100)
