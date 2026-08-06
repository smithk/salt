"""Bad input must fail with a message, not a traceback.

SALT is meant to be pointed at arbitrary files, so the common mistakes —
wrong path, wrong format, truncated download, a spreadsheet saved as the
wrong thing — are ordinary inputs, not exceptional ones. Each must produce
one clear line naming the file and the problem.

Fixtures are generated here rather than committed. A file whose entire
purpose is to be malformed is cheaper to write in three lines than to store,
review, and carry in git forever, and generating it documents exactly what
is wrong with it.
"""

import os

import pandas as pd
import pytest

from saltml.cli import main
from saltml.data import DatasetError, load


def write(path, content):
    """Write bytes or text and return the path."""
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as handle:
        handle.write(content)
    return path


# --- files that are broken in various ways -------------------------------

def test_empty_file(tmp_path):
    path = write(tmp_path / "empty.csv", "")
    with pytest.raises(DatasetError, match="empty"):
        load(path)


def test_empty_arff(tmp_path):
    path = write(tmp_path / "empty.arff", "")
    with pytest.raises(DatasetError, match="empty"):
        load(path)


def test_not_a_data_file_at_all(tmp_path):
    path = write(tmp_path / "notes.arff", "# just some prose\nnothing structured here\n")
    with pytest.raises(DatasetError):
        load(path)


def test_binary_content(tmp_path):
    path = write(tmp_path / "image.arff", bytes(range(256)) * 8)
    with pytest.raises(DatasetError, match="binary|not text"):
        load(path)


def test_arff_header_without_data_section(tmp_path):
    path = write(
        tmp_path / "truncated.arff",
        "@relation truncated\n@attribute a numeric\n@attribute b numeric\n",
    )
    with pytest.raises(DatasetError):
        load(path)


def test_arff_with_declared_but_absent_rows(tmp_path):
    path = write(
        tmp_path / "norows.arff",
        "@relation empty\n@attribute a numeric\n@attribute y {p,q}\n@data\n",
    )
    with pytest.raises(DatasetError, match="no rows|ended"):
        load(path)


def test_csv_with_only_a_header(tmp_path):
    path = write(tmp_path / "headers.csv", "a,b,target\n")
    with pytest.raises(DatasetError, match="no rows"):
        load(path)


def test_csv_with_ragged_rows(tmp_path):
    path = write(tmp_path / "ragged.csv", "a,b,y\n1,2,x\n3,4,5,6,7,y\n")
    with pytest.raises(DatasetError):
        load(path)


def test_single_column_file_has_no_features(tmp_path):
    path = write(tmp_path / "one.csv", "y\np\nq\np\n")
    with pytest.raises(ValueError, match="no feature columns"):
        load(path)


def test_missing_file_is_distinct_from_malformed(tmp_path):
    # A wrong path and a corrupt file are different mistakes and must not be
    # reported the same way.
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "never_existed.csv")


def test_target_column_entirely_missing(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "y": [None, None, None]})
    with pytest.raises(ValueError):
        load(frame)


def test_constant_target_is_refused(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": ["p", "p", "p", "p"]})
    with pytest.raises(ValueError, match="distinct value"):
        load(frame)


# --- errors must be legible ----------------------------------------------

def test_message_names_the_file_and_the_problem(tmp_path):
    path = write(tmp_path / "broken.arff", "not an arff file\n")
    with pytest.raises(DatasetError) as caught:
        load(path)
    message = str(caught.value)
    assert "broken.arff" in message
    assert len(message) > 30, f"unhelpfully terse: {message!r}"


def test_no_bare_stopiteration_escapes(tmp_path):
    """scipy raises StopIteration with no message; that must never reach a user."""
    path = write(tmp_path / "header_only.arff", "@relation r\n@attribute a numeric\n")
    with pytest.raises(DatasetError) as caught:
        load(path)
    assert str(caught.value).strip(), "error carried no message at all"


def test_unknown_extension_warns_and_tries_csv(tmp_path, caplog):
    import logging

    path = write(tmp_path / "data.weird", "a,b,y\n1,2,p\n3,4,q\n5,6,p\n")
    with caplog.at_level(logging.WARNING, logger="saltml"):
        dataset = load(path)
    assert dataset.n_samples == 3
    assert "Unrecognised extension" in caplog.text


# --- the CLI must not show a traceback -----------------------------------

@pytest.mark.parametrize(
    "name,content",
    [
        ("empty.csv", ""),
        ("prose.arff", "this is not arff\n"),
        ("binary.arff", bytes(range(256))),
        ("headers.csv", "a,b,target\n"),
    ],
)
def test_cli_reports_bad_input_cleanly(tmp_path, capsys, name, content):
    path = write(tmp_path / name, content)
    code = main(["fit", str(path), "--trials", "1", "--quiet"])
    assert code == 2, "bad input should exit 2, not crash"
    err = capsys.readouterr().err
    assert err.startswith("error:") or "error:" in err
    assert "Traceback" not in err


def test_cli_reports_a_missing_file_cleanly(tmp_path, capsys):
    code = main(["fit", str(tmp_path / "nope.csv"), "--trials", "1", "--quiet"])
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_is_reported_not_swallowed(tmp_path, capsys):
    path = write(tmp_path / "locked.csv", "a,b,y\n1,2,p\n")
    path.chmod(0o000)
    try:
        code = main(["fit", str(path), "--trials", "1", "--quiet"])
        assert code == 2
        assert "Traceback" not in capsys.readouterr().err
    finally:
        path.chmod(0o644)
