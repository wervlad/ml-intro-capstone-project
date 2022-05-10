from click.testing import CliRunner
import joblib
from pathlib import Path
import numpy as np
import pytest
import forestml.data as data
from forestml.train import train
from .test_data import generate_test_dataset


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


def test_train_fails_with_invalid_logreg_max_iter_value(
    runner: CliRunner,
) -> None:
    """
    It fails when passing to LogisticRegression invalid maximum number of
    iterations.
    """
    ret = runner.invoke(
        train, ["--search", "manual", "logreg", "--max-iter", -1]
    )
    assert ret.exit_code == 2
    assert "Invalid value for '--max-iter'" in ret.output


def test_train_fails_with_invalid_logreg_C_value(runner: CliRunner) -> None:
    """It fails when passing to LogisticRegression invalid C value."""
    ret = runner.invoke(train, ["--search", "manual", "logreg", "--c", 0])
    assert ret.exit_code == 2
    assert "Invalid value for '--c'" in ret.output


def test_train_fails_with_invalid_knn_n_neighbors_value(
    runner: CliRunner,
) -> None:
    """It fails when passing to KNN invalid number of neighbors."""
    ret = runner.invoke(
        train, ["--search", "manual", "knn", "--n-neighbors", 0]
    )
    assert ret.exit_code == 2
    assert "Invalid value for '--n-neighbors'" in ret.output


def test_train_fails_with_invalid_knn_metric_value(runner: CliRunner) -> None:
    """It fails when passing to KNN invalid metric value."""
    ret = runner.invoke(
        train, ["--search", "manual", "knn", "--metric", "foo"]
    )
    assert ret.exit_code == 2
    assert "Invalid value for '--metric'" in ret.output


def test_train_fails_with_invalid_knn_weights_value(runner: CliRunner) -> None:
    """It fails when passing to KNN invalid weights value."""
    ret = runner.invoke(
        train, ["--search", "manual", "knn", "--weights", "foo"]
    )
    assert ret.exit_code == 2
    assert "Invalid value for '--weights'" in ret.output


def test_train_manual_search_generates_model_successfully(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        generate_test_dataset()
        model_path = Path(data.MODEL_PATH)
        assert not model_path.is_file()
        ret = runner.invoke(
            train,
            [
                "--use-scaler",
                True,
                "--transform",
                None,
                "--search",
                "manual",
                "logreg",
                "--max-iter",
                1000,
                "--c",
                0.001,
            ],
        )
        assert ret.exit_code == 0
        assert model_path.is_file()
        assert is_saved_model_correct(model_path)


def test_train_random_search_generates_model_successfully(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        generate_test_dataset()
        model_path = Path(data.MODEL_PATH)
        assert not model_path.is_file()
        ret = runner.invoke(
            train,
            [
                "--use-scaler",
                True,
                "--transform",
                None,
                "--search",
                "random",
                "logreg",
            ],
        )
        assert ret.exit_code == 0
        assert model_path.is_file()
        assert is_saved_model_correct(model_path)


def is_saved_model_correct(path: Path):
    X, _ = data.get_dataset(data.DATASET_PATH)
    X = X.to_numpy()
    obj = joblib.load(path)
    y_pred = obj.predict(X)
    return np.logical_and(y_pred >= 0, y_pred <= 6).all()
