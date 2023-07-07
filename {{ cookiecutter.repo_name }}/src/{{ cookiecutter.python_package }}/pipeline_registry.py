"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from {{ cookiecutter.python_package }}.pipelines.iris import (
    create_end_to_end_pipeline,
    create_data_engineering_pipeline,
    create_data_science_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = create_end_to_end_pipeline()
    pipelines["data_science"] = create_data_science_pipeline()
    pipelines["data_engineering"] = create_data_engineering_pipeline()
    pipelines["end_to_end"] = create_end_to_end_pipeline()
    return pipelines
