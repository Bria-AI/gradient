import numpy as np
import torch
from torchvision.transforms import v2
from utils.image_utils.supported_ratios import (
    ratio_configs_256,
    ratio_configs_512,
    ratio_configs_1024,
)
from anypathlib import AnyPath
import os
import cv2
from multiprocessing import Pool

MAX_CPU_COUNT = os.cpu_count() or 1

if os.environ.get("DEBUG", False):
    MAX_CPU_COUNT = 1


def get_new_size(image: np.array, resolution: int, ratio: float):
    if resolution == 256:
        ratio_configs = ratio_configs_256
    elif resolution == 512:
        ratio_configs = ratio_configs_512
    elif resolution == 1024:
        ratio_configs = ratio_configs_1024
    else:
        raise ValueError("Invalid resolution")

    if ratio not in ratio_configs:
        raise ValueError("Invalid ratio")
    w, h = image.size

    config = ratio_configs[ratio]
    ratio = max(config["height"] / h, config["width"] / w)
    new_w, new_h = int(np.ceil(w * ratio)), int(np.ceil(h * ratio))
    return new_w, new_h


def generate_transform(height: int, width: int, center_crop: bool = True):
    """
    Generate a transform for resizing an image to a specific resolution and ratio.
    """
    transform_params = [v2.ToDtype(torch.uint8, scale=True), v2.Resize((height, width))]
    if center_crop:
        transform_params.append(v2.CenterCrop((height, width)))

    transform = v2.Compose(transform_params)
    return transform


def resize_image(image: np.array, transformation: v2.Compose):
    """
    Resize an image to a specific resolution and ratio.
    """
    image = torch.from_numpy(image).permute(2, 0, 1)
    return transformation(image)


def files_in_dir_generator(dir_path: str):
    """
    Generate all files in a directory.
    """
    path = AnyPath(dir_path)
    assert path.is_dir(), "Invalid directory path"
    for file in path.iterdir():
        if file.is_file():
            yield file


def resize_image_file(
    file: AnyPath, transformation: v2.Compose, output_path: AnyPath = None
):
    """
    Resize an image file to a specific resolution and ratio.
    """
    try:
        image = cv2.imread(str(file))
        image = resize_image(image, transformation)
        cv2.imwrite(str(output_path), image)
    except Exception as e:
        print(f"Error resizing image {file}: {e}")
        raise e


def resize_full_dir(
    dir_path: str, resolution: int, output_dir: str, center_crop: bool = False
):
    """
    Resize all images in a directory to a specific resolution and ratio.
    """
    image_dir = AnyPath(dir_path)
    output_dir = AnyPath(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        raise e
    assert image_dir.is_dir(), "Invalid directory path"
    files = image_dir.iterdir()
    transform = generate_transform(resolution, center_crop)
    with Pool(MAX_CPU_COUNT) as p:
        p.starmap(resize_image_file, [(file, transform) for file in files])
