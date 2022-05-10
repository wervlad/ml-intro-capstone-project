from click.testing import CliRunner
import pytest
from forestml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_train_fails_with_invalid_number_of_splits(runner: CliRunner) -> None:
    """It fails when number of splits for KFold/Nested CV is less than 2."""
    ret = runner.invoke(train, ["--n-splits", 0])
    assert ret.exit_code == 2
    assert "Invalid value for '--n-splits'" in ret.output

def test_train_fails_with_invalid_transform_value(runner: CliRunner) -> None:
    """It fails when passing invalid value for transform option."""
    ret = runner.invoke(train, ["--transform", "foo"])
    assert ret.exit_code == 2
    assert "Invalid value for '--transform'" in ret.output

def test_train_fails_with_invalid_search_value(runner: CliRunner) -> None:
    """It fails when passing invalid value for search option."""
    ret = runner.invoke(train, ["--search", "foo"])
    assert ret.exit_code == 2
    assert "Invalid value for '--search'" in ret.output

def test_train_fails_with_invalid_random_state(runner: CliRunner) -> None:
    """It fails when passing invalid value for random_state option."""
    ret = runner.invoke(train, ["--random-state", -1])
    assert ret.exit_code == 2
    assert "Invalid value for '--random-state'" in ret.output
    ret = runner.invoke(train, ["--random-state", 2**32])
    assert ret.exit_code == 2
    assert "Invalid value for '--random-state'" in ret.output

def test_train_fails_with_invalid_logreg_max_iter_value(runner: CliRunner) -> None:
    """
    It fails when passing to LogisticRegression invalid maximum number of
    iterations.
    """
    ret = runner.invoke(train, ["--search", "manual", "logreg", "--max-iter", -1])
    assert ret.exit_code == 2
    assert "Invalid value for '--max-iter'" in ret.output

def test_train_fails_with_invalid_logreg_C_value(runner: CliRunner) -> None:
    """It fails when passing to LogisticRegression invalid C value."""
    ret = runner.invoke(train, ["--search", "manual", "logreg", "--c", 0])
    assert ret.exit_code == 2
    assert "Invalid value for '--c'" in ret.output

def test_train_fails_with_invalid_knn_n_neighbors_value(runner: CliRunner) -> None:
    """It fails when passing to KNN invalid number of neighbors."""
    ret = runner.invoke(train, ["--search", "manual", "knn", "--n-neighbors", 0])
    assert ret.exit_code == 2
    assert "Invalid value for '--n-neighbors'" in ret.output

def test_train_fails_with_invalid_knn_metric_value(runner: CliRunner) -> None:
    """It fails when passing to KNN invalid metric value."""
    ret = runner.invoke(train, ["--search", "manual", "knn", "--metric", "foo"])
    assert ret.exit_code == 2
    assert "Invalid value for '--metric'" in ret.output

def test_train_fails_with_invalid_knn_weights_value(runner: CliRunner) -> None:
    """It fails when passing to KNN invalid weights value."""
    ret = runner.invoke(train, ["--search", "manual", "knn", "--weights", "foo"])
    assert ret.exit_code == 2
    assert "Invalid value for '--weights'" in ret.output
