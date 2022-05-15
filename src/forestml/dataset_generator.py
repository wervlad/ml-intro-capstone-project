"""
download dataset from https://archive.ics.uci.edu/ml/datasets/Covertype
and put it to data/covtype.data
"""
import click
import pandas as pd
from pathlib import Path

ORIGINAL_DATASET_PATH = "data/covtype.data"
TEST_TRUE_PATH = "data/test_true.csv"
COLUMNS = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1",
    "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1",
    "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6",
    "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11",
    "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16",
    "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21",
    "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26",
    "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31",
    "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36",
    "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40", "Cover_Type"
]
FIRST_COL = [2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5]

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=ORIGINAL_DATASET_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-d",
    "--test-true-path",
    default=TEST_TRUE_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def generate_test_true(dataset_path: Path, test_true_path: Path):
    df = pd.read_csv(dataset_path)
    df.index.rename("Id", inplace=True)
    df.columns = COLUMNS
    df.loc[-1] = FIRST_COL
    df.index += 2
    df.sort_index(inplace=True)
    # df.to_csv("data/covtype.csv")  # full dataset
    # df.loc[:15120].to_csv("data/train.csv")  # train
    # df.drop("Cover_Type", axis=1).loc[15121:].to_csv("data/test.csv")  # test
    df.loc[15121:]["Cover_Type"].to_csv(test_true_path)  # test true labels

if __name__ == "__main__":
    generate_test_true(ORIGINAL_DATASET_PATH)
