import click
import numpy as np
import mlflow
import mlflow.sklearn
from math import inf
from joblib import dump
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import get_dataset, DATASET_PATH, MODEL_PATH
import warnings

# Ignore tSNE FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


@click.group()
@click.pass_context
@click.option(
    "-d",
    "--dataset-path",
    default=DATASET_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default=MODEL_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--random-state",
    default=42,
    show_default=True,
    type=click.IntRange(0, 2**32 - 1),
)
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
@click.option(
    "--search",
    default="random",
    show_default=True,
    type=click.Choice(["manual", "random"]),
)
def train(
    ctx: click.core.Context,
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_splits: int,
    use_scaler: bool,
    transform: str,
    search: str,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["dataset_path"] = dataset_path
    ctx.obj["save_model_path"] = save_model_path
    ctx.obj["random_state"] = random_state
    ctx.obj["n_splits"] = n_splits
    ctx.obj["use_scaler"] = use_scaler
    ctx.obj["transform"] = transform
    ctx.obj["search"] = search


@train.command()
@click.pass_context
@click.option(
    "--max-iter", default=1000, show_default=True, type=click.IntRange(0, inf)
)
@click.option(
    "--c",
    default=1.0,
    show_default=True,
    type=click.FloatRange(0, inf, min_open=True),
)
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
    if ctx.obj["search"] == "manual":
        run_experiment(
            ctx,
            create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
        )
    else:
        run_experiment_random_grid(
            ctx,
            create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
        )


@train.command()
@click.pass_context
@click.option(
    "--n-neighbors",
    default=5,
    show_default=True,
    type=click.IntRange(0, inf, min_open=True),
)
@click.option(
    "--metric",
    default="minkowski",
    show_default=True,
    type=click.Choice(
        [
            "euclidean",
            "manhattan",
            "chebyshev",
            "minkowski",
            "wminkowski",
            "seuclidean",
            "mahalanobis",
        ]
    ),
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
    weights: str,
) -> None:
    ctx.obj["model"] = "KNN"
    ctx.obj["n_neighbors"] = n_neighbors
    ctx.obj["metric"] = metric
    ctx.obj["weights"] = weights
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric=metric, weights=weights
    )
    if ctx.obj["search"] == "manual":
        run_experiment(
            ctx,
            create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
        )
    else:
        run_experiment_random_grid(
            ctx,
            create_pipeline(use_scaler=ctx.obj["use_scaler"], model=model),
        )


def create_pipeline(use_scaler: bool, model: BaseEstimator) -> Pipeline:
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
        X, y = get_dataset(ctx.obj["dataset_path"])
        X = X.to_numpy()
        y = y.to_numpy()
        if ctx.obj["transform"] == "tsne":
            X = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                random_state=42,
            ).fit_transform(X)
        elif ctx.obj["transform"] == "lda":
            X = LinearDiscriminantAnalysis(
                n_components=2,
                priors=None,
                shrinkage="auto",
                solver="eigen",
                store_covariance=False,
                tol=0.0001,
            ).fit_transform(X, y)
        kf = KFold(
            n_splits=ctx.obj["n_splits"],
            shuffle=True,
            random_state=ctx.obj["random_state"],
        )
        # use KFold cross-validator to calculate metrics
        for train_indices, test_indices in kf.split(X, y):
            X_train = X[train_indices]
            X_val = X[test_indices]
            y_train = y[train_indices]
            y_val = y[test_indices]
            pipeline.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, pipeline.predict(X_val))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val,
                pipeline.predict(X_val),
                average="weighted",
                zero_division=0,
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
        mlflow.log_param("search", "KFold (manual)")
        mlflow.log_param("use_scaler", ctx.obj["use_scaler"])
        mlflow.log_param("transform", ctx.obj["transform"])
        mlflow.log_metric("accuracy", np.mean(accuracy_list))
        mlflow.log_metric("precision", np.mean(precision_list))
        mlflow.log_metric("recall", np.mean(recall_list))
        mlflow.log_metric("f1", np.mean(f1_list))
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1: {f1}.")
        dump(pipeline, ctx.obj["save_model_path"])
        click.echo(f"Model is saved to {ctx.obj['save_model_path']}.")


def run_experiment_random_grid(
    ctx: click.core.Context,
    pipeline: Pipeline,
) -> None:
    with mlflow.start_run():
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        best_params_list = []
        X, y = get_dataset(ctx.obj["dataset_path"])
        X = X.to_numpy()
        y = y.to_numpy()
        if ctx.obj["transform"] == "tsne":
            X = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                random_state=42,
            ).fit_transform(X)
        elif ctx.obj["transform"] == "lda":
            X = LinearDiscriminantAnalysis(
                n_components=2,
                priors=None,
                shrinkage="auto",
                solver="eigen",
                store_covariance=False,
                tol=0.0001,
            ).fit_transform(X, y)
        cv_outer = KFold(
            n_splits=10, shuffle=True, random_state=ctx.obj["random_state"]
        )
        if ctx.obj["model"] == "LogReg":
            PARAMS = {
                "classifier__max_iter": [1000],
                "classifier__C": [
                    0.00001,
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1,
                    10,
                    1000,
                    10000,
                ],
            }
        elif ctx.obj["model"] == "KNN":
            PARAMS = {
                "classifier__n_neighbors": np.logspace(
                    0,
                    3,
                    num=100,
                    dtype=int,
                ),
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": [
                    "euclidean",
                    "manhattan",
                    "chebyshev",
                    "minkowski",
                ],
            }
        click.echo("Running Nested CV with RandomSearch in inner loop")
        for train_indices, test_indices in cv_outer.split(X):
            X_train, X_test = X[train_indices, :], X[test_indices, :]
            y_train, y_test = y[train_indices], y[test_indices]
            cv_inner = KFold(
                n_splits=3, shuffle=True, random_state=ctx.obj["random_state"]
            )
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=PARAMS,
                n_iter=5,
                scoring="accuracy",
                n_jobs=-1,
                cv=cv_inner,
                refit=True,
            )
            best_model = random_search.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, best_model.predict(X_test))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                best_model.predict(X_test),
                average="weighted",
                zero_division=0,
            )
            click.echo(f"{accuracy}, {best_model.best_params_}")
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            best_params_list.append(best_model.best_params_)
        mlflow.log_param("model", ctx.obj["model"])
        best_of_the_best = best_params_list[np.array(accuracy_list).argmax()]
        click.echo(f"Best params: {best_of_the_best}")
        if ctx.obj["model"] == "LogReg":
            mlflow.log_param(
                "max_iter",
                best_of_the_best["classifier__max_iter"],
            )
            mlflow.log_param("logreg_c", best_of_the_best["classifier__C"])
        elif ctx.obj["model"] == "KNN":
            mlflow.log_param(
                "n_neighbors",
                best_of_the_best["classifier__n_neighbors"],
            )
            mlflow.log_param("metric", best_of_the_best["classifier__metric"])
            mlflow.log_param(
                "weights",
                best_of_the_best["classifier__weights"],
            )
        mlflow.log_param("search", "NestedCV (random)")
        mlflow.log_param("use_scaler", ctx.obj["use_scaler"])
        mlflow.log_param("transform", ctx.obj["transform"])
        mlflow.log_metric("accuracy", np.mean(accuracy_list))
        mlflow.log_metric("precision", np.mean(precision_list))
        mlflow.log_metric("recall", np.mean(recall_list))
        mlflow.log_metric("f1", np.mean(f1_list))
        click.echo(
            f"Accuracy: {np.mean(accuracy_list):.3f} "
            f"(std: {np.std(accuracy_list):.3f})"
        )
        click.echo(
            f"Precision: {np.mean(precision_list):.3f} "
            f"(std: {np.std(precision_list):.3f})"
        )
        click.echo(
            f"Recall: {np.mean(recall_list):.3f} "
            f"(std: {np.std(recall_list):.3f})"
        )
        click.echo(f"F1: {np.mean(f1_list):.3f} (std: {np.std(f1_list):.3f})")
        # finally fit best model on whole dataset and save it
        pipeline.set_params(**best_of_the_best)
        pipeline.fit(X, y)
        dump(pipeline, ctx.obj["save_model_path"])
        click.echo(f"Model is saved to {ctx.obj['save_model_path']}.")
        # summarize the estimated performance of the model
