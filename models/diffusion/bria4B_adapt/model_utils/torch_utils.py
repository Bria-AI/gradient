import glob
from typing import List

import numpy as np
import webdataset as wds


def get_params(model):
    model_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params[name] = param

    return model_params


def load_dataset_from_tars(
    training_dirs: List[str],
    rank: int,
    world_size: int,
    seed: int = 0,
    slice: bool = False,
):
    tar_files = []
    for train_data_dir in training_dirs:
        files = glob.glob(f"{train_data_dir}/*.tar")
        print(f"We have {len(files)} data files on {train_data_dir}")
        tar_files += files

    total = len(tar_files)

    print(f"We have {total} data files in total")

    if slice:
        print("Using slicing")
        tar_files = tar_files[
            rank::world_size
        ]  # starting from rank skip world size and take object at each step

        print(f"Process {rank} will use data files {len(tar_files)} files")

    np.random.seed(seed)
    np.random.shuffle(tar_files)

    def duplicate_name_exception_handler(e):
        if type(e) == ValueError:
            return True  # happens on duplicate vhashes
        raise e

    train_dataset = wds.WebDataset(
        tar_files,
        nodesplitter=wds.split_by_worker,
        handler=duplicate_name_exception_handler,
        shardshuffle=True,
    ).decode("torch")

    return train_dataset
