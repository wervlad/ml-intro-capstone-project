Homework for RS School Machine Learning course.

This project uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for predicting forest cover type.
1. Clone this repository to your machine.
2. Download Forest train dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and Poetry are installed on your machine (I use Poetry 1.1.13).
4. Run train with the following command:
```bash
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options in CLI. To get a full list of them, use help:
```bash
poetry run train --help
```
