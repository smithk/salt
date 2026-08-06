"""Benchmark suite definitions and the fetch cache.

The suite tables are hand-written OpenML ids, which is exactly the kind of
thing that rots silently — a wrong id fetches the wrong dataset rather than
failing. The structural tests here run offline; the ones that talk to OpenML
are marked ``slow`` so a normal run stays fast and network-free.
"""

import pytest

from saltml.benchmarks import SUITES, Suite, cache_dir, resolve_suite
from saltml.cli import main
from saltml.task import Task


def test_every_suite_has_entries_and_a_reference():
    for suite in SUITES.values():
        assert suite.entries, f"{suite.name} is empty"
        assert suite.description
        assert suite.reference, f"{suite.name} must cite its source"


def test_openml_ids_are_unique_within_a_suite():
    for suite in SUITES.values():
        ids = [e.openml_id for e in suite.entries]
        assert len(ids) == len(set(ids)), f"{suite.name} repeats an OpenML id"


def test_entry_names_are_unique_within_a_suite():
    for suite in SUITES.values():
        names = [e.name for e in suite.entries]
        assert len(names) == len(set(names))


def test_declared_tasks_are_real():
    for suite in SUITES.values():
        for entry in suite.entries:
            assert entry.task in (Task.CLASSIFICATION, Task.REGRESSION)


def test_regression_suite_is_all_regression():
    assert SUITES["ctr23-lite"].tasks == {Task.REGRESSION}


def test_classification_suite_is_all_classification():
    assert SUITES["cc18-lite"].tasks == {Task.CLASSIFICATION}


def test_for_task_filters():
    suite = SUITES["smoke"]
    assert all(e.task is Task.REGRESSION for e in suite.for_task(Task.REGRESSION))
    assert len(suite.for_task(None)) == len(suite.entries)


def test_resolve_suite_names_the_alternatives():
    with pytest.raises(KeyError, match="cc18-lite"):
        resolve_suite("nope")


def test_cache_is_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("SALTML_CACHE", str(tmp_path / "somewhere"))
    assert cache_dir() == tmp_path / "somewhere"


def test_cache_defaults_under_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("SALTML_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_dir() == tmp_path / "saltml" / "benchmarks"


def test_bench_list_runs(capsys):
    assert main(["bench", "list"]) == 0
    out = capsys.readouterr().out
    assert "cc18-lite" in out and "ctr23-lite" in out
    assert "cache:" in out


@pytest.mark.slow
def test_fetch_round_trips_a_real_dataset(tmp_path, monkeypatch):
    """Fetch one small dataset and load it back as a Dataset."""
    monkeypatch.setenv("SALTML_CACHE", str(tmp_path))
    from saltml.benchmarks import SUITES, load_entry

    entry = next(e for e in SUITES["smoke"].entries if e.name == "iris")
    dataset = load_entry(entry)
    assert dataset.task is Task.CLASSIFICATION
    assert dataset.n_samples == 150
    assert dataset.n_features == 4
    # Second call must hit the cache, not the network.
    assert load_entry(entry).n_samples == 150


@pytest.mark.slow
def test_fetched_regression_dataset_keeps_its_task(tmp_path, monkeypatch):
    monkeypatch.setenv("SALTML_CACHE", str(tmp_path))
    from saltml.benchmarks import SUITES, load_entry

    entry = next(e for e in SUITES["smoke"].entries if e.name == "autoPrice")
    assert load_entry(entry).task is Task.REGRESSION
