import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import hydra
import typer
from omegaconf import OmegaConf

from solarnet.utils.log import init_log, set_log_level
from solarnet.utils.yaml import load_yaml

init_log()

from solarnet.tasks.test import test
from solarnet.tasks.train import train_standard
from solarnet.tasks.test_on_two import test_new

set_log_level(logging.WARNING)
logger = logging.getLogger()
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def read_config(parameters_overrides: Optional[List[str]]) -> dict:
    # Initialize configuration (using Hydra)
    hydra.initialize(config_path="./../config")
    config = hydra.compose(config_name="config", overrides=parameters_overrides)
    return OmegaConf.to_container(config, resolve=True)


@app.command("train")
def train_command(
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose:
        set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    training_type = config["training_type"]
    if training_type == "train":
        train_standard(config)
    elif training_type == "ssl":
        train_ssl(config)
    elif training_type == "finetune":
        finetune(config)


@app.command("test")
def test_command(
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose:
        set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    test(config, verbose)


@app.command("test_multi")
def test_command(
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose:
        set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    test_new(config, verbose)









class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"







# Command to add options before the command (-v train ...)
@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    if verbose:
        set_log_level(logging.INFO)


if __name__ == "__main__":
    app()
