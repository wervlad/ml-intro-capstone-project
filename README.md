<a href="https://github.com/wervlad/9_evaluation_selection/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/wervlad/9_evaluation_selection.svg?color=blue">
</a>
<a href="https://github.com/wervlad/9_evaluation_selection/actions/workflows/black.yml">
    <img alt="Code style: black" src="https://github.com/wervlad/9_evaluation_selection/actions/workflows/black.yml/badge.svg">
</a>
<a href="https://github.com/wervlad/9_evaluation_selection/actions/workflows/flake8.yml">
    <img alt="flake8" src="https://github.com/wervlad/9_evaluation_selection/actions/workflows/flake8.yml/badge.svg">
</a>
<a href="https://github.com/wervlad/9_evaluation_selection/actions/workflows/mypy.yml">
    <img alt="mypy" src="https://github.com/wervlad/9_evaluation_selection/actions/workflows/mypy.yml/badge.svg">
</a>
<a href="https://github.com/wervlad/9_evaluation_selection/actions/workflows/tests.yml">
    <img alt="mypy" src="https://github.com/wervlad/9_evaluation_selection/actions/workflows/tests.yml/badge.svg">
</a>

<br>

# Homework for RS School Machine Learning course

This project uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for predicting forest cover type.

**I strongly recommend using this package in a Unix environment. Its functionality under Windows is not guaranteed. I use Debian 11.**
1. Clone this repository to your machine.
2. Download [Forest train](https://github.com/wervlad/data/raw/master/train.csv.zip) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model> <ml model>
```
E.g. if you saved dataset to data/train.csv, the following command will perform hyperparameters search for KNN algorithm and the save best model to data/model.joblib:
```sh
poetry run train knn
```
Currently only KNN and Linear Regression algorithm are supported.

If you'd like to specify hyperparams manually, use ``manual`` value in ``--search`` argument. E.g. the following command will standardize data and then train and save Logistic Regression model with regularization strenght (C) 10, maximum number of iterations 1000:
```sh
run train --use-scaler True --transform None --search manual logreg --max-iter=1000 --c 10
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
To get supported hyperparameters run:
```sh
poetry run train <ml model> --help
```
E.g.
```sh
poetry run train knn --help
poetry run train logreg --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
You can then visit http://localhost:5000 in your browser and investigate results of all experiments. Typical log will look like this:

![MLFlow experiments example](https://github.com/wervlad/9_evaluation_selection/blob/main/img/experiments.png)

Note: I highlighted the one with the best score with red frame manually. You'll not see it in your browser.

7. For better undestanding of dataset automatic EDA report can be generated:
```sh
poetry run profiling_report -i <path to csv with data> -o <path to save report>
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```
Currently all tests are passing.

![tests passed successfully](https://github.com/wervlad/9_evaluation_selection/blob/main/img/tests.png)
