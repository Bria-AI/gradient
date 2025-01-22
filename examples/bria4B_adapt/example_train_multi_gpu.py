import os

from accelerate.commands.launch import launch_command

CONFIG_FILE_PATH = os.getcwd() + "/core/config/default_config.yaml"
TRAIN_FILE_PATH = os.getcwd() + "/examples/bria4B_adapt/example_train.py"
args = [
    "--config_file",
    CONFIG_FILE_PATH,
    TRAIN_FILE_PATH,
]
launch_command(args)
