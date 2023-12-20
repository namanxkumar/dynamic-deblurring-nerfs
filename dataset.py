import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from einops import rearrange

import random
import os
import torch
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from imageio import imread


class Adobe240Singular(Dataset):
    def __init__(
        self,
        blur_directory: str,
        sharp_directory: str,
        num_frames_per_blur: int = 7,
        image_extension: str = "png",
        downscale_factor: int = 4,
    ):
        self.blur_directory = blur_directory
        self.sharp_directory = sharp_directory
        self.num_frames_per_blur = num_frames_per_blur
        self.image_extension = image_extension
        self.filename_format = "{:05d}"

        extract_index = lambda x: int(osp.basename(x).split(".")[0])

        self.blur_image_paths = sorted(
            glob(osp.join(self.blur_directory, f"*.{self.image_extension}")),
            key=extract_index,
        )

        self.downscale_factor = downscale_factor
        self.image_height = imread(self.blur_image_paths[0]).shape[0] // self.downscale_factor
        self.image_width = imread(self.blur_image_paths[0]).shape[1] // self.downscale_factor

        self.sharp_image_paths = [
            [
                osp.join(
                    self.sharp_directory,
                    self.filename_format.format(i) + "." + self.image_extension,
                )
                for i in range(
                    extract_index(blur_image_path)
                    - ((self.num_frames_per_blur) // 2),
                    extract_index(blur_image_path)
                    + ((self.num_frames_per_blur) // 2)
                    + 1,
                )
            ]
            for blur_image_path in self.blur_image_paths
        ]

    def __len__(self):
        return len(self.blur_image_paths)

    def __getitem__(self, index):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.image_height // self.downscale_factor, self.image_width // self.downscale_factor)
                ),
            ]
        )
        sample = {}
        sample["blur_image"] = transform(imread(self.blur_image_paths[index]))
        sample["sharp_images"] = torch.stack(
            [transform(imread(path)) for path in self.sharp_image_paths[index]],
            dim=0,
        )
        sample["time_step"] = (index * self.num_frames_per_blur) + (self.num_frames_per_blur // 2)
        return sample

    def visualize(self, index):
        print("blur_image_path:\n", self.blur_image_paths[index])
        print("sharp_image_paths:\n", self.sharp_image_paths[index])
        blur_image = np.array(imread(self.blur_image_paths[index]))
        plt.figure(figsize=(4 * blur_image.shape[1] // blur_image.shape[0], 4))
        plt.axis("off")
        plt.imshow(blur_image)
        plt.show()
        sharp_images = [
            np.array(imread(path)) for path in self.sharp_image_paths[index]
        ]
        sharp_images = np.concatenate(sharp_images, axis=1)
        plt.figure(figsize=(4 * (sharp_images.shape[1] // sharp_images.shape[0]), 4))
        plt.axis("off")
        plt.imshow(sharp_images)
        plt.show()
