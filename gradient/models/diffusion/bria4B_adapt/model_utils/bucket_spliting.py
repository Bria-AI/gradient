import glob
import json
import os
from pprint import pprint
from typing import Dict, List

import numpy as np

from .db_data_channels import (
    data_channels_flux_256_precompute,
    data_channels_flux_512_precompute,
)


def get_bucket_index_stochastic(rank, bucket_dirs: List[str]):

    # Consider allocation the first len(bucket_dirs) to all the buckets before randomization
    # bucket_dirs = sorted(bucket_dirs)
    amounts = []
    for bucket in bucket_dirs:
        amount = len(glob.glob(f"{bucket}/*"))
        amounts += [amount]

    total = np.sum(amounts)
    probs = [amount / total for amount in amounts]

    if rank == 0:
        print("---------Data Distribution-------")
        data = {}
        for bucket, amount in zip(bucket_dirs, amounts):
            data[os.path.basename(bucket)] = f"{amount/total:.4f}"

        pprint(data)

    # Random - set seed according to rank is called before
    i = np.random.choice(range(len(bucket_dirs)), 1, p=probs)[0]

    return i


def get_area(p):
    p = os.path.basename(p)
    azure_input_extenstion = "INPUT_"
    p = p.replace(azure_input_extenstion, "")
    _, ratio, _, width, _, height = p.split("_")
    ratio = ratio
    width = int(width)
    height = int(height)
    return width * height


def get_deterministic_training_dirs_dynamic_batches(
    rank: int,
    world_size: int,
    bucket_dirs: List[str],
    dense_caption_ratio: float,
    dynamic_batches: Dict[int, int],
):
    # dynamic_batches {256 * 256: 32, ....}

    # get relavant pieces
    areas = set(
        [
            min(dynamic_batches.keys(), key=lambda a: abs(get_area(b) - a))
            for b in bucket_dirs
        ]
    )

    dynamic_batches = {area: dynamic_batches[area] for area in areas}

    # More gpu's to lower batches
    total_batch = sum(dynamic_batches.values())
    probs = {area: total_batch / dynamic_batches[area] for area in dynamic_batches}
    probs = {area: probs[area] / sum(probs.values()) for area in probs}

    occupied_gpus = 0
    for i, area in enumerate(dynamic_batches):
        if i == len(dynamic_batches) - 1:
            world_size_piece_ = world_size - occupied_gpus
        else:
            world_size_piece_ = int(probs[area] * world_size)
        bucket_dirs_piece = [
            b
            for b in bucket_dirs
            if min(dynamic_batches.keys(), key=lambda a: abs(get_area(b) - a)) == area
        ]

        if rank < occupied_gpus + world_size_piece_:
            return get_deterministic_training_dirs(
                rank - occupied_gpus,
                world_size_piece_,
                bucket_dirs_piece,
                dense_caption_ratio,
            )

        occupied_gpus += world_size_piece_


def get_deterministic_training_dirs(rank, world_size, bucket_dirs, dense_caption_ratio):
    caption_training_dirs = [f"{t}/captions" for t in bucket_dirs]
    dense_caption_training_dirs = [f"{t}/dense_captions" for t in bucket_dirs]

    # We give the first gpu's to dense and other to non-dense
    dense_gpus_world = int(dense_caption_ratio * world_size)
    non_dense_gpus_world = world_size - dense_gpus_world

    if rank < dense_gpus_world:
        bucket_index = get_bucket_index(
            rank, dense_gpus_world, dense_caption_training_dirs
        )
        return dense_caption_training_dirs[bucket_index]

    bucket_index = get_bucket_index(
        rank - dense_gpus_world, non_dense_gpus_world, caption_training_dirs
    )
    return caption_training_dirs[bucket_index]


# Default Bucketing Scheme
def get_bucket_index(rank, world_size, bucket_dirs: List[str]):

    amounts = []
    for bucket in bucket_dirs:
        amount = len(glob.glob(f"{bucket}/*"))
        amounts += [amount]

    total = np.sum(amounts)
    probs = [amount / total for amount in amounts]

    if rank == 0:
        print("---------Data Distribution-------")
        data = {}
        for bucket, amount in zip(bucket_dirs, amounts):
            data[bucket] = f"{amount/total:.4f}"

        # pprint(data)

    # Determinsitic
    allocations = [
        int(p * world_size) for p in probs
    ]  # This sum will always be <=world_size since int rounds down

    # Make sure all gpu's are working
    while sum(allocations) < world_size:
        ind_min = np.argmin(allocations)
        allocations[ind_min] += 1

    # Fill empty buckets so that all buckets would participateif possible
    if world_size >= len(bucket_dirs):
        while min(allocations) == 0:
            ind_min = np.argmin(allocations)
            ind_max = np.argmax(allocations)
            allocations[ind_max] -= 1
            allocations[ind_min] += 1

    allocated_gpus = 0

    # example: allocations=[1,0,2]-> rank 0->  bucket 0, rank 1-> bucket 2, rank 2-> bucket 2
    for i, amount in enumerate(allocations):
        allocated_gpus += amount
        if rank < allocated_gpus:
            return i

    # Left overs will be alocated to around (1k,1k)
    leftover_amount = world_size - allocated_gpus  #  8%20 = 8
    left = int(len(bucket_dirs) / 2 - leftover_amount / 2)
    ind = left + (rank - allocated_gpus) % leftover_amount

    # Just in case :)
    ind = np.clip(ind, 0, len(bucket_dirs) - 1)

    return ind


# 1k,1k settings for curation
def get_bucket_index_curated(rank, world_size, bucket_dirs: List[str]):
    bucket_dict = {
        0: "ratio_068_",
        1: "ratio_078_",
        2: "ratio_138_",
        3: "ratio_146_",
        4: "ratio_129_",
        5: "ratio_068_",
        6: "ratio_138_",
        7: "ratio_146_",
        8: "ratio_146_",
        9: "ratio_129_",
        10: "ratio_129_",
        11: "ratio_129_",
        12: "ratio_1_",
        13: "ratio_072_",
        14: "ratio_072_",
        15: "ratio_121_",
    }

    assert world_size == len(bucket_dict)

    for i, bucket in enumerate(bucket_dirs):
        if bucket_dict[rank] in bucket:
            return i

    if rank == 0:
        print(bucket_dirs)

    assert False, f"rank {rank} not found in bucket_dict"


# 1k,1k settings for curation
"""
d={
'ratio_069': 264,
'ratio_074': 236,
'ratio_081': 113,
'ratio_1': 90,
'ratio_124': 94,
'ratio_129': 66,
'ratio_135': 440,
'ratio_145': 376
}
"""


def get_bucket_index_curated_hr(rank, world_size, bucket_dirs: List[str]):
    return 0
    bucket_dict = {
        0: "ratio_069_",
        1: "ratio_069_",
        2: "ratio_074_",
        3: "ratio_074_",
        4: "ratio_081_",
        5: "ratio_1_",
        6: "ratio_124_",
        7: "ratio_129_",
        8: "ratio_135_",
        9: "ratio_135_",
        10: "ratio_135_",
        11: "ratio_135_",
        12: "ratio_145_",
        13: "ratio_145_",
        14: "ratio_145_",
        15: "ratio_145_",
    }

    assert world_size == len(bucket_dict)

    for i, bucket in enumerate(bucket_dirs):
        if bucket_dict[rank] in bucket:
            return i

    if rank == 0:
        print(bucket_dirs)

    assert False, f"rank {rank} not found in bucket_dict"


if __name__ == "__main__":
    # Simulation
    # buckets = glob.glob('/home/ubuntu/eiga/precompute_webdataset_test/*')
    # buckets = glob.glob('/home/ubuntu/eiga/precompute_curated_webdataset/*')

    buckets = glob.glob("/home/ubuntu/eiga/datasets/flux_ratio_*/*")
    buckets = list(sorted(buckets))

    dynamic_batches = {256 * 256: 32, 512 * 512: 12, 1024 * 1024: 3}

    allocation = {}
    for world_size in [32]:
        print(f"------------{world_size}-----------")
        for ind in range(world_size):
            train_dir = dir = get_deterministic_training_dirs_dynamic_batches(
                ind,
                world_size,
                buckets,
                dense_caption_ratio=0.8,
                dynamic_batches=dynamic_batches,
            )
            allocation[ind] = train_dir
            print(ind, allocation[ind])

    with open("/home/ubuntu/spring/buckets.json", "w") as f:
        json.dump(allocation, f)
        # print('----Allocated-----')
        # pprint(allocation)

        # data={os.path.basename(b):0 for b in buckets}
        # for buckets_ind in allocation.values():
        #     data[os.path.basename(buckets[buckets_ind])]+=1

        # pprint(data)
        # for all in allocation.values():
        #     pprint(buckets[all])
