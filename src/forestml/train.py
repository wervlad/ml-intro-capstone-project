from pathlib import Path
from joblib import dump
import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option("--random-state", default=42, show_default=True, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
@click.option(
    "--use-scaler",
    default=True,
    show_default=True,
    type=bool,
)
@click.option("--max-iter", default=100, show_default=True, type=int)
@click.option("--logreg-c", default=1.0, show_default=True, type=float)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_val, pipeline.predict(features_val), average="weighted"
        )
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1: {f1}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
