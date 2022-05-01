from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(use_scaler: bool, random_state: int) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("classifier", LogisticRegression(random_state=random_state)))
    return Pipeline(steps=pipeline_steps)
