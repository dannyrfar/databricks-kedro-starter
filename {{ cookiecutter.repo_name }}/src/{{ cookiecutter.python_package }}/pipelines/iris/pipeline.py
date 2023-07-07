from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_features, generate_predictions


def create_end_to_end_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(lambda x: x, "iris_csv", "iris_bronze", name="raw_to_bronze"),
            node(generate_features, "iris_bronze", "iris_features", name="features"),
            node(
                generate_predictions,
                "iris_features",
                ["iris_predictions", "regression_model", "accuracy"],
                name="train_and_predictions",
            ),
        ]
    )


def create_data_engineering_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(lambda x: x, "iris_csv", "iris_bronze", name="raw_to_bronze"),
            node(generate_features, "iris_bronze", "iris_features", name="features"),
        ]
    )


def create_data_science_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                generate_predictions,
                "iris_features",
                ["iris_predictions", "regression_model", "accuracy"],
                name="train_and_predictions",
            ),
        ]
    )
