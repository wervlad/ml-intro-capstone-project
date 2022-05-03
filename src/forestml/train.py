from math import inf
from pathlib import Path
from joblib import dump
import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
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
    "--n-splits",
    default=5,
    show_default=True,
    type=click.IntRange(2, inf),
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
    n_splits: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    with mlflow.start_run():
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        X, y = get_dataset(dataset_path)
        X = X.to_numpy()
        y = y.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        # use KFold cross-validator to calculate metrics
        for train_indices, test_indices in kf.split(X, y):
            X_train = X[train_indices]
            X_val = X[test_indices]
            y_train = y[train_indices]
            y_val = y[test_indices]
            pipeline.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, pipeline.predict(X_val))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, pipeline.predict(X_val), average="weighted"
            )
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        pipeline.fit(X, y)  # finally train model on whole dataset
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", sum(accuracy_list) / len(accuracy_list))
        mlflow.log_metric("precision", sum(precision_list) / len(precision_list))
        mlflow.log_metric("recall", sum(recall_list) / len(recall_list))
        mlflow.log_metric("f1", sum(f1_list) / len(f1_list))
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1: {f1}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
