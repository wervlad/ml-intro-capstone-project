import logging
from pathlib import Path
from typing import Tuple, Union

import click
import pandas as pd
import pandas_profiling

DATASET_PATH = "data/train.csv"
REPORT_PATH = "data/report.html"
TARGET = "Cover_Type"
DROP = ["Soil_Type7", "Soil_Type15"]


def get_dataset(
    csv_path: Path,
    return_X_y: bool = True,
    drop_columns: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    if drop_columns:
        dataset.drop(DROP, axis=1, inplace=True)
    if return_X_y:
        features = dataset.drop(TARGET, axis=1)
        target = dataset[TARGET]
        return features, target
    else:
        return dataset


@click.command()
@click.option(
    "-i",
    "--csv-path",
    default=DATASET_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--report-path",
    default=REPORT_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def generate_profiling_report(csv_path: Path, report_path: Path) -> None:
    profile = pandas_profiling.ProfileReport(
        get_dataset(csv_path, return_X_y=False, drop_columns=False), minimal=False
    )
    profile.to_file(report_path)
