import webdataset as wds
import pandas as pd
import os
from PIL import Image
from tempfile import NamedTemporaryFile
from pathlib import Path


def read_image_from_file(image_path: Path):
    """
    Convert an image to png.
    """
    if image_path.suffix == ".png":
        with open(image_path, "rb") as f:
            return f.read()
    image = Image.open(image_path)
    with NamedTemporaryFile(delete=False, suffix=".png") as img_file:
        image.save(img_file, format="png")
        with open(img_file.name, "rb") as f:
            return f.read()


def get_metadata_file(image_dir, metadata_file_name):
    metadata_file = Path.joinpath(image_dir, metadata_file_name)
    assert metadata_file.is_file(), "Invalid metadata file path"
    if metadata_file.name.endswith(".csv"):
        df = pd.read_csv(metadata_file.name)
    elif metadata_file.name.endswith(".json"):
        df = pd.read_json(metadata_file.name)
    elif metadata_file.name.endswith(".jsonl"):
        df = pd.read_json(metadata_file, lines=True)
    else:
        raise ValueError("Invalid metadata file format")
    return df


def simple_png_wds_packer(
    input_dir: str,
    output_dir: str,
    metadata_file_name: str = "metadata.csv",
    caption_col: str = "caption",
    image_col: str = "image",
):
    """
    Pack images in a directory into a WebDataset with the image path as key, image as png file and the txt.
    image_dir: str - The directory containing the images and the csv.
    output_dir: str - The directory to save the WebDataset.
    csv_file_name: str - The name of the csv file containing the image path and their captions.
    caption_col: str - The column name containing the captions.
    image_col: str - The column name containing the image paths.
    """
    image_dir = Path(input_dir)
    output_dir = Path(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        raise e
    assert image_dir.is_dir(), "Invalid directory path"
    df = get_metadata_file(image_dir, metadata_file_name)
    urls = df[image_col].tolist()
    captions = df[caption_col].tolist()
    full_paths = [image_dir / url for url in urls]
    sink = wds.ShardWriter(str(output_dir / "dataset-%06d.tar"))
    for image_path, caption in zip(full_paths, captions):
        try:
            image = read_image_from_file(image_path)
            sample = {"__key__": image_path.name, "png": image, "txt": caption}
            sink.write(sample)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
    sink.close()
    return output_dir
