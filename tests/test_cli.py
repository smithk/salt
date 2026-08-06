import argparse

import pytest

from saltml.cli import main, parse_duration

IRIS = "data/classification/iris.arff"


@pytest.mark.parametrize(
    "text,seconds",
    [("90", 90), ("30s", 30), ("10m", 600), ("1h", 3600), ("1.5m", 90)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "10d", "-5"])
def test_parse_duration_rejects_nonsense(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_duration(text)


def test_learners_listing_runs(capsys):
    assert main(["learners"]) == 0
    out = capsys.readouterr().out
    assert "classification" in out and "regression" in out
    assert "random_forest" in out


def test_fit_writes_a_model(tmp_path, capsys):
    destination = tmp_path / "model.joblib"
    code = main(
        ["fit", IRIS, "--trials", "6", "--folds", "3", "--jobs", "1",
         "--top", "3", "-o", str(destination), "--quiet"]
    )
    assert code == 0
    assert destination.exists()
    assert "best:" in capsys.readouterr().out


def test_missing_file_exits_cleanly(capsys):
    assert main(["fit", "/nonexistent/nope.csv", "--trials", "1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_learner_exits_cleanly(capsys):
    code = main(["fit", IRIS, "--trials", "1", "--learners", "banana", "--quiet"])
    assert code == 2
    assert "banana" in capsys.readouterr().err
