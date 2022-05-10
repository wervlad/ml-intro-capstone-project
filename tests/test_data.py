from click.testing import CliRunner
from pathlib import Path
from sklearn.datasets import make_classification
import pytest
import pandas as pd
import numpy as np
import forestml.data as data

N_SAMPLES = 100
N_FEATURES = 55


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def generate_test_dataset(
    path: Path = Path(data.DATASET_PATH),
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    random_state: int = 42,
) -> None:
    """Helper function to generate dataset for tests."""
    # generate data randomly
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        n_informative=45,
        n_redundant=10,
        n_classes=7,
    )
    data = np.concatenate((X, y.reshape(1, -1).T), axis=1)
    columns = [""] * 56
    columns[-1] = "Cover_Type"
    columns[0] = "Id"
    for i in range(15, 55):
        columns[i] = f"Soil_Type{i-14}"
    # wrap data into dataset and save it to file
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=data, columns=columns).set_index("Id").to_csv(path)


def test_get_dataset_fails_with_invalid_path(runner: CliRunner) -> None:
    """It fails when trying to get dataset from unexistent file."""
    with runner.isolated_filesystem():
        with pytest.raises(FileNotFoundError):
            data.get_dataset(Path("invalid_path/invalid_dataset.css"))


def test_get_dataset_with_return_X_y_returns_correct_shape(
    runner: CliRunner,
) -> None:
    """Checks the shape of loaded dataset is correct with return_X_y=True."""
    with runner.isolated_filesystem():
        path = Path(data.DATASET_PATH)
        generate_test_dataset(path, n_samples=N_SAMPLES, n_features=N_FEATURES)
        dataset = data.get_dataset(path, return_X_y=True, drop_columns=False)
        assert len(dataset) == 2
        assert dataset[0].shape == (N_SAMPLES, N_FEATURES)
        assert dataset[1].shape == (N_SAMPLES,)


def test_get_dataset_wo_return_X_y_returns_correct_shape(
    runner: CliRunner,
) -> None:
    """Checks the shape of loaded dataset is correct with return_X_y=False."""
    with runner.isolated_filesystem():
        path = Path(data.DATASET_PATH)
        generate_test_dataset(path, n_samples=N_SAMPLES, n_features=N_FEATURES)
        dataset = data.get_dataset(path, return_X_y=False, drop_columns=False)
        assert dataset.shape == (N_SAMPLES, N_FEATURES + 1)


def test_get_dataset_with_drop_columns_returns_correct_shape(
    runner: CliRunner,
) -> None:
    """
    Checks the shape of loaded dataset is correct after droping constant
    columns.
    """
    with runner.isolated_filesystem():
        path = Path(data.DATASET_PATH)
        generate_test_dataset(path, n_samples=N_SAMPLES, n_features=N_FEATURES)
        dataset = data.get_dataset(path, return_X_y=False, drop_columns=True)
        assert dataset.shape == (N_SAMPLES, N_FEATURES + 1 - len(data.DROP))


def test_generate_profiling_report_generates_report(runner: CliRunner) -> None:
    """Checks the profile report is generated successfully."""
    with runner.isolated_filesystem():
        path = Path(data.DATASET_PATH)
        # generate empty dataset to speedup profiling
        generate_test_dataset(path, n_samples=0, n_features=N_FEATURES)
        assert not Path(data.REPORT_PATH).is_file()
        ret = runner.invoke(data.generate_profiling_report)
        assert ret.exit_code == 0
        assert Path(data.REPORT_PATH).is_file()
