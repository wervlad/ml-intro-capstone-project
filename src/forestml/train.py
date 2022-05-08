import click
import mlflow
import mlflow.sklearn
import pandas as pd
from math import inf
from joblib import dump
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from .data import get_dataset, DATASET_PATH
import warnings

# Ignore tSNE FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

@click.group()
@click.pass_context
@click.option(
    "-d",
    "--dataset-path",
    default=DATASET_PATH,
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
@click.option(
    "--transform",
    default=None,
    show_default=True,
    type=click.Choice(["lda", "tsne", "None"]),
)
def train(
    ctx: click.core.Context,
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_splits: int,
    use_scaler: bool,
    transform: str,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["X"], ctx.obj["y"] = get_dataset(dataset_path)
    ctx.obj["save_model_path"] = save_model_path
    ctx.obj["random_state"] = random_state
    ctx.obj["n_splits"] = n_splits
    ctx.obj["use_scaler"] = use_scaler
    ctx.obj["transform"] = transform

@train.command()
@click.pass_context
@click.option("--max-iter", default=100, show_default=True, type=int)
@click.option("--c", default=1.0, show_default=True, type=float)
def logreg(
    ctx: click.core.Context,
    max_iter: int,
    c: float,
) -> None:
    ctx.obj["model"] = "LogReg"
    ctx.obj["max_iter"] = max_iter
    ctx.obj["c"] = c
    model = LogisticRegression(
        random_state=ctx.obj["random_state"], max_iter=max_iter, C=c
    )
    run_experiment(
        ctx,
        create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
    )

@train.command()
@click.pass_context
@click.option("--n-neighbors", default=5, show_default=True, type=int)
@click.option(
    "--metric",
    default="minkowski",
    show_default=True,
    type=click.Choice(["euclidean", "manhattan", "chebyshev", "minkowski",
                       "wminkowski", "seuclidean", "mahalanobis"]),
)
@click.option(
    "--weights",
    default="uniform",
    show_default=True,
    type=click.Choice(["uniform", "distance"]),
)
def knn(
    ctx: click.core.Context,
    n_neighbors: int,
    metric: str,
    weights:str,
) -> None:
    ctx.obj["model"] = "KNN"
    ctx.obj["n_neighbors"] = n_neighbors
    ctx.obj["metric"] = metric
    ctx.obj["weights"] = weights
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric=metric, weights=weights
    )
    run_experiment(
        ctx,
        create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
    )

def create_pipeline(
    use_scaler: bool, model: BaseEstimator
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("classifier", model))
    return Pipeline(steps=pipeline_steps)

def run_experiment(
    ctx: click.core.Context,
    pipeline: Pipeline,
) -> None:
    with mlflow.start_run():
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        X = ctx.obj["X"].to_numpy()
        y = ctx.obj["y"].to_numpy()
        if ctx.obj["transform"] == "tsne":
            X = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                random_state=42
            ).fit_transform(X)
        elif ctx.obj["transform"] == "lda":
            X = LinearDiscriminantAnalysis(
                n_components=2,
                priors=None,
                shrinkage='auto',
                solver='eigen',
                store_covariance=False,
                tol=0.0001
            ).fit_transform(X, y)
        kf = KFold(n_splits=ctx.obj["n_splits"],
                   shuffle=True,
                   random_state=ctx.obj["random_state"])
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
        mlflow.log_param("model", ctx.obj["model"])
        if ctx.obj["model"] == "LogReg":
            mlflow.log_param("max_iter", ctx.obj["max_iter"])
            mlflow.log_param("logreg_c", ctx.obj["c"])
        elif ctx.obj["model"] == "KNN":
            mlflow.log_param("n_neighbors", ctx.obj["n_neighbors"])
            mlflow.log_param("metric", ctx.obj["metric"])
            mlflow.log_param("weights", ctx.obj["weights"])
        mlflow.log_param("use_scaler", ctx.obj["use_scaler"])
        mlflow.log_param("transform", ctx.obj["transform"])
        mlflow.log_metric("accuracy", sum(accuracy_list) / len(accuracy_list))
        mlflow.log_metric("precision", sum(precision_list) / len(precision_list))
        mlflow.log_metric("recall", sum(recall_list) / len(recall_list))
        mlflow.log_metric("f1", sum(f1_list) / len(f1_list))
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1: {f1}.")
        dump(pipeline, ctx.obj["save_model_path"])
        click.echo(f"Model is saved to {ctx.obj['save_model_path']}.")
