import http.client
import json
import logging
import os
import sys
from typing import Any, Dict

import git
import mlflow
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, DataSetError
from kedro.pipeline import Pipeline
from mlflow.entities import RunStatus

MLFLOW_MAX_PARAM_LEN = 250

logger = logging.getLogger(__name__)


class MLFlowHooks:
    """
    Initialise mlflow and log useful tracking info automatically
    """

    @hook_impl
    def after_catalog_created(self, conf_creds, feed_dict):
        """
        Enables bearer token based authentication to mlflow service, secured
        behind SSO authentication.

        Args:
            conf_creds: kedro credentials config
            feed_dict: kedro feed dict

        Returns:

        """
        parameters = feed_dict["parameters"]

        # MLFlow enabled?
        self.mlflow_enabled = parameters.get("mlflow_enabled", False)
        if not self.mlflow_enabled:
            return

        # Change to authentication based on server_type
        if parameters.get("mlflow_instance") == "databricks":
            logger.info("Authneticating Databricks MLFlow")
            mlflow_creds = conf_creds.get("mlflow")
            set_databricks_creds(mlflow_creds)

        # MLFlow SSO enabled?
        if not parameters.get("mlflow_sso_enabled", False):
            return

        # set mlflow tracking token
        conn = http.client.HTTPSConnection(
            conf_creds["mlflow_client_credentials"]["oauth_domain"]
        )

        payload = {
            "client_id": conf_creds["mlflow_client_credentials"]["client_id"],
            "client_secret": conf_creds["mlflow_client_credentials"]["client_secret"],
            "audience": conf_creds["mlflow_client_credentials"]["audience"],
            "grant_type": conf_creds["mlflow_client_credentials"]["grant_type"],
        }

        headers = {"content-type": "application/json"}

        conn.request("POST", "/oauth/token", json.dumps(payload), headers)

        res = conn.getresponse()
        data = res.read()

        token_dict = json.loads(data.decode("utf-8"))
        os.environ["MLFLOW_TRACKING_TOKEN"] = token_dict["access_token"]

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """
        Initialise mlflow tracking and experiment

        Args:
            run_params: kedro run parameters
            pipeline: kedro pipeline
            catalog: kedro catalog

        Returns:

        """
        if not self.mlflow_enabled:
            return

        tracking_uri = catalog.datasets.params__mlflow_tracking_uri.load()
        registry_uri = catalog.datasets.params__mlflow_tracking_uri.load()
        experiment = catalog.datasets.params__mlflow_experiment.load()

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        mlflow.set_experiment(experiment)

        should_init_mlflow = False

        for ds in pipeline.all_outputs():
            try:
                cls_name = catalog._get_dataset(ds).__class__.__name__
            except DataSetError:
                continue  # ignore MemoryDataSets

            if "mlflow" in cls_name.lower():
                should_init_mlflow = True
                # mlflow.spark.autolog()
                break

        if should_init_mlflow:
            mlflow.start_run()
            _log_kedro_info(run_params, pipeline, catalog)
            _log_git_sha()

    @hook_impl
    def after_pipeline_run(self):
        if not self.mlflow_enabled:
            return

        if mlflow.active_run() is not None:
            mlflow.end_run()

    @hook_impl
    def on_pipeline_error(self):
        if not self.mlflow_enabled:
            return

        if mlflow.active_run() is not None:
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))


def _log_kedro_info(
    run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
) -> None:
    """
    Logs useful kedro info in `kedro.yml` artifact file under the run
    """

    # this will have all the nested structures (duplicates)
    parameters_artifacts = catalog._data_sets["parameters"].load()

    # similar to context.params
    extra_params = run_params.get("extra_params")
    if extra_params is not None:
        parameters_artifacts.update(extra_params)
    parameters_artifacts = {
        _sanitise_kedro_param(param_name): param_value
        for param_name, param_value in parameters_artifacts.items()
    }

    mlflow_log_params = parameters_artifacts.get("mlflow_log_params")
    if mlflow_log_params:
        kedro_parameters_filtered = {
            k: v for k, v in parameters_artifacts.items() if k in mlflow_log_params
        }
    else:
        kedro_parameters_filtered = parameters_artifacts
    _write_params(kedro_parameters_filtered)

    mlflow.log_dict(
        {
            # could use `run_params` in the future
            "kedro_run_args": None
            if os.environ.get("DATABRICKS_RUNTIME_VERSION") is not None
            else " ".join(repr(a) if " " in a else a for a in sys.argv[1:]),
            "kedro_nodes": sorted(n.short_name for n in pipeline.nodes),
            "kedro_dataset_versions": list(_get_dataset_versions(catalog, pipeline)),
            "kedro_parameters": parameters_artifacts,
        },
        "kedro.yml",
    )


def _log_git_sha():
    """
    Logs git-sha to mlflow commit parameter
    """
    mlflow.set_tag("mlflow.source.git.commit", get_git_sha())


def _sanitise_kedro_param(param_name):
    """
    Removes disallowed special characters
    """
    sanitised_param_name = param_name.replace(":", "_")
    return sanitised_param_name


def _get_dataset_versions(catalog: DataCatalog, pipeline: Pipeline):
    """
    Logs version info for versioned kedro datasets
    """
    for ds_name, ds in sorted(catalog._data_sets.items()):
        ds_in_out = ds_name in pipeline.all_outputs()
        try:
            save_ver = ds.resolve_save_version() if ds_in_out else None
            load_ver = ds.resolve_save_version() if ds_in_out else None
        except AttributeError:
            save_ver = None
            load_ver = None
        if save_ver or load_ver:
            version_info = {
                "name": ds_name,
                "save_version": save_ver,
                "load_version": load_ver,
            }
            yield version_info


def get_git_sha():
    try:
        git_repo = git.Repo(search_parent_directories=True)
        return git_repo.head.object.hexsha
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.error("Unable to get 'git_sha'")
        logger.error(str(exc))
    return None


def _write_params(params: dict):
    params = {k: str(v)[:MLFLOW_MAX_PARAM_LEN] for k, v in params.items()}
    len_params = len(params)
    if len_params > 100:
        dict_items = list(params.items())
        for i in range(0, len_params + 1, 100):
            batch = dict(dict_items[i : i + 100])
            mlflow.log_params(batch)
    else:
        mlflow.log_params(params)


def set_databricks_creds(mlflow_creds):
    """
    Pass databricks credentials as OS variables.
    """
    # https://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html
    # TODO: What if CLI token is wrong?
    if mlflow_creds is not None and (
        os.environ.get("DATABRICKS_HOST") is None
        or os.environ.get("DATABRICKS_TOKEN") is None
    ):
        os.environ["DATABRICKS_HOST"] = os.environ.get(
            "DATABRICKS_HOST", mlflow_creds["DATABRICKS_HOST"]
        )
        os.environ["DATABRICKS_TOKEN"] = os.environ.get(
            "DATABRICKS_TOKEN", mlflow_creds["DATABRICKS_TOKEN"]
        )
