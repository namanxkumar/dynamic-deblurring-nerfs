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
        offset: int = 0,
        num_frames: int = None,
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

        if num_frames is not None:
            self.blur_image_paths = self.blur_image_paths[offset:offset + num_frames]

        self.downscale_factor = downscale_factor
        self.image_height = (
            imread(self.blur_image_paths[0]).shape[0] // self.downscale_factor
        )
        self.image_width = (
            imread(self.blur_image_paths[0]).shape[1] // self.downscale_factor
        )

        self.sharp_image_paths = [
            [
                osp.join(
                    self.sharp_directory,
                    self.filename_format.format(i) + "." + self.image_extension,
                )
                for i in range(
                    extract_index(blur_image_path) - ((self.num_frames_per_blur) // 2),
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
                    (
                        self.image_height // self.downscale_factor,
                        self.image_width // self.downscale_factor,
                    )
                ),
            ]
        )
        sample = {}
        sample["blur_image"] = transform(imread(self.blur_image_paths[index]))
        sample["sharp_images"] = torch.stack(
            [transform(imread(path)) for path in self.sharp_image_paths[index]],
            dim=0,
        )
        sample["time_step"] = (index * self.num_frames_per_blur) + (
            self.num_frames_per_blur // 2
        )
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


class Adobe240SingularPixels(Dataset):
    def __init__(
        self,
        blur_directory: str,
        sharp_directory: str,
        num_frames_per_blur: int = 7,
        image_extension: str = "png",
        downscale_factor: int = 4,
        num_chunks: int = 1,
        offset: int = 0,
        num_frames: int = None,
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

        if num_frames is not None:
            self.blur_image_paths = self.blur_image_paths[offset:offset + num_frames]

        self.downscale_factor = downscale_factor
        self.image_height = (
            imread(self.blur_image_paths[0]).shape[0] // self.downscale_factor
        )
        self.image_width = (
            imread(self.blur_image_paths[0]).shape[1] // self.downscale_factor
        )
        self.num_pixels = self.image_height * self.image_width

        self.num_chunks = num_chunks
        self.num_pixels_per_chunk = self.num_pixels // self.num_chunks

        self.sharp_image_paths = [
            [
                osp.join(
                    self.sharp_directory,
                    self.filename_format.format(i) + "." + self.image_extension,
                )
                for i in range(
                    extract_index(blur_image_path) - ((self.num_frames_per_blur) // 2),
                    extract_index(blur_image_path)
                    + ((self.num_frames_per_blur) // 2)
                    + 1,
                )
            ]
            for blur_image_path in self.blur_image_paths
        ]

    def __len__(self):
        return len(self.blur_image_paths)

    @staticmethod
    def _chunk_pixel_indices(pixel_indices, num_pixels, num_chunks, chunk_index):
        pixel_indices = pixel_indices[
            chunk_index
            * (num_pixels // num_chunks) : (chunk_index + 1)
            * (num_pixels // num_chunks)
        ]
        return pixel_indices

    def _pixel_indices_to_xy(self, pixel_indices):
        pixel_x = pixel_indices % self.image_width
        pixel_y = pixel_indices // self.image_width
        return pixel_x, pixel_y

    def __getitem__(self, index):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (
                        self.image_height,
                        self.image_width,
                    ),
                    antialias=True,
                ),
            ]
        )

        pixel_indices = torch.randperm(self.num_pixels)
        stack_indices = [
            self._chunk_pixel_indices(
                pixel_indices, self.num_pixels, self.num_chunks, chunk_index
            )
            for chunk_index in range(self.num_chunks)
        ]
        stack_indices_xy = [
            torch.stack(
                self._pixel_indices_to_xy(indices),
                dim=-1,
            )
            for indices in stack_indices
        ]

        sample = {}
        sample["pixel_indices"] = torch.stack(
            stack_indices_xy, dim=0
        )  # Shape: (num_chunks, num_pixels_per_chunk, 2)

        blur_image = transform(imread(self.blur_image_paths[index])).view(-1, 3)

        stack_image = [
            blur_image[stack_indices[chunk_index]]
            for chunk_index in range(self.num_chunks)
        ]
        sample["blur_image"] = torch.stack(
            stack_image, dim=0
        )  # Shape: (num_chunks, num_pixels_per_chunk, 3)

        sharp_images = torch.stack(
            [
                transform(imread(path)).view(-1, 3)
                for path in self.sharp_image_paths[index]
            ],
            dim=0,
        )
        stack_sharp_images = [
            sharp_images[:, stack_indices[chunk_index]]
            for chunk_index in range(self.num_chunks)
        ]
        sample["sharp_images"] = torch.stack(
            stack_sharp_images, dim=0
        )  # Shape: (num_chunks, num_frames_per_blur, num_pixels_per_chunk, 3)

        sample["time_step"] = (index * self.num_frames_per_blur) + (
            self.num_frames_per_blur // 2
        )  # Shape: (1)
        return sample
