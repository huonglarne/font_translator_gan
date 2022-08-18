from pathlib import Path
import pickle

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

from src.constants import (
    IMG_SIZE,
    STYLE_CHANNEL,
    TRAIN_PATH,
)


def get_pickle_file_list(folder: Path, outfile: Path):
    assert outfile.suffix == ".pkl"

    if outfile.exists():
        with open(outfile, "rb") as f:
            return pickle.load(f)
    else:
        file_list = sorted(folder.glob("*/*"))
        with open(outfile, "wb") as f:
            pickle.dump(file_list, f)
        return file_list


class FontDataset(Dataset):
    def __init__(
        self,
        subset_folder: Path = TRAIN_PATH,
        style_language: str = "english",
        gt_language: str = "chinese",
        style_channel: int = STYLE_CHANNEL,
        img_size: int = IMG_SIZE,
    ):
        self.gt_folder = subset_folder / gt_language
        self.style_folder = subset_folder / style_language
        self.content_folder = subset_folder / "source"

        self.gt_list = get_pickle_file_list(
            self.gt_folder, self.gt_folder / "list_files.pkl"
        )

        self.style_channel = style_channel
        self.img_size = img_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
        )

    def __getitem__(self, index):
        gt_img_path = self.gt_list[index]

        font_name = gt_img_path.parent.name
        style_img_paths = list((self.style_folder / font_name).glob("*"))[:6]

        character_filename = gt_img_path.name
        content_img_path = self.content_folder / character_filename

        content_image = self.load_image(content_img_path)
        gt_image = self.load_image(gt_img_path)
        style_image = torch.cat(
            [self.load_image(style_path) for style_path in style_img_paths], 0
        )

        return {
            "gt_images": gt_image,
            "content_images": content_image,
            "style_images": style_image,
            "style_image_paths": [str(each) for each in style_img_paths],
            "image_paths": str(gt_img_path),
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.gt_list)

    def load_image(self, path):
        image = Image.open(path)
        image = self.transform(image)
        return image
