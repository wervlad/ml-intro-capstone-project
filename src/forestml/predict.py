from pathlib import Path
import joblib
import click
from .data import get_dataset, MODEL_PATH, TARGET

TEST_DATASET_PATH = "data/test.csv"
TEST_W_LABELS_DATASET_PATH = "data/covtype.data"
SAMPLE_SUBMISSION_PATH = "data/sampleSubmission.csv"
SUBMISSION_PATH = "data/submission.csv"


@click.command()
@click.option(
    "-t",
    "--test-path",
    default=TEST_DATASET_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--submission-path",
    default=SUBMISSION_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default=MODEL_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def predict(test_path: Path, submission_path: Path, model_path: Path) -> None:
    """Generate prediction for Kaggle."""
    X = get_dataset(Path(test_path), return_X_y=False).to_numpy()
    model = joblib.load(model_path)
    submission = get_dataset(
        Path(SAMPLE_SUBMISSION_PATH), return_X_y=False, drop_columns=False
    )
    submission[TARGET] = model.predict(X)
    submission.to_csv(SUBMISSION_PATH, index=False)
