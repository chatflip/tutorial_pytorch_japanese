import glob
import os
import shutil

import hydra
import mlflow
import yaml
from mlflow import pytorch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig


class MlflowWriter:
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception:
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name
            ).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, "models")

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, timestamp=None, step=None):
        self.client.log_metric(self.run_id, key, value, timestamp, step)

    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def move_mlruns(self):
        # runのコピー
        hydra_cwd = os.getcwd()
        exp_root, exp_id = os.path.split(hydra_cwd)
        src_mlrun_dir = os.path.join(hydra_cwd, "mlruns", "1")
        src_mlrun_path = [
            file_folder
            for file_folder in glob.glob(f"{src_mlrun_dir}/*")
            if os.path.isdir(file_folder)
        ][0]
        run_hash = os.path.basename(src_mlrun_path)
        dst_mlrun_path = os.path.join(
            hydra.utils.get_original_cwd(), "mlruns", "1", run_hash
        )
        shutil.copytree(src_mlrun_path, dst_mlrun_path)
        overwrite_meta_yaml(dst_mlrun_path, run_hash)
        # experimentのコピー
        dst_exp_path = os.path.join(hydra.utils.get_original_cwd(), "mlruns", "1")
        copy_exp_meta_yaml(src_mlrun_dir, dst_exp_path)


def overwrite_meta_yaml(run_path, run_hash):
    yaml_path = os.path.join(run_path, "meta.yaml")
    with open(yaml_path, "r") as yml:
        config = yaml.safe_load(yml)
    config["artifact_uri"] = f"mlruns/1/{run_hash}/artifacts"
    with open(yaml_path, "w") as file:
        yaml.dump(config, file)


def copy_exp_meta_yaml(src_exp_path, dst_exp_path):
    src_path = os.path.join(src_exp_path, "meta.yaml")
    with open(src_path, "r") as yml:
        config = yaml.safe_load(yml)
    config["artifact_location"] = "./"
    dst_path = os.path.join(dst_exp_path, "meta.yaml")
    with open(dst_path, "w") as file:
        yaml.dump(config, file)
