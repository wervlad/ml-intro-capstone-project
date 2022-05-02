Homework for RS School Machine Learning course.

This project uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for predicting forest cover type.
1. Clone this repository to your machine.
2. Download [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies:
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
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
