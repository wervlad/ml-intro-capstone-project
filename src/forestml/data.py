import logging
from pathlib import Path
from typing import Tuple

import click
import pandas as pd

DATASET_PATH = "data/train.csv"
TARGET = "Cover_Type"

def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(TARGET, axis=1)
    target = dataset[TARGET]
    return features, target
