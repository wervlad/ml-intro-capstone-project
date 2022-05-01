from pathlib import Path
import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def train(dataset_path: Path) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    classifier = LogisticRegression(random_state=42).fit(features_train, target_train)
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
