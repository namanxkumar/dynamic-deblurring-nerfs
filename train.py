from dataset import Adobe240Singular
from model.model import Model
from losses import blur_combination_loss

from accelerate import Accelerator

from einops import rearrange

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from typing import Dict, List, Tuple, Union

from tqdm import tqdm

from pathlib import Path


def exists(val):
    return val is not None

class Step:
    def __init__(self, step: int, batch_size: int):
        self.step = step
        self.batch_size = batch_size

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["step"]
        self.batch_size = (
            state_dict["batch_size"] if "batch_size" in state_dict else self.batch_size
        )

    def state_dict(self):
        state_dict = {
            "step": self.step,
            "batch_size": self.batch_size,
        }
        return state_dict

class Trainer:
    def __init__(
        self,
        batch_size: int = 4,
        num_steps: int = 10000,
        num_steps_per_save: int = 1000,
        learning_rate: float = 1e-4,
        inject_function = None,
        blur_directory: str = "data/adobe240singulardataset/train_blur/GOPR9647/",
        sharp_directory: str = "data/adobe240singulardataset/train/GOPR9647/",
        results_directory: str = "results/",
        image_extension: str = "png",
        image_downscale_factor: int = 1,
        num_time_steps_per_frame: int = 7,
        num_coarse_samples_per_ray: int = 64,
        num_fine_samples_per_ray: int = 128,
        near_depth: float = 0.1,
        far_depth: float = 10.0,
        num_scene_trajectory_basis_coefficients: int = 24,
        num_camera_trajectory_basis_coefficients: int = 24,
        num_voxels_per_axis: int = 128,
        min_bound_per_axis: float = 1.0,
        max_bound_per_axis: float = 10.0,
        voxel_dim: int = 16,
        color_model_hidden_dim: int = 64,
    ):
        self.num_steps = num_steps
        self.num_steps_per_save = num_steps_per_save

        self.inject_function = inject_function
        
        self.results_directory = Path(results_directory)

        self.accelerator = Accelerator()

        self.dataset = Adobe240Singular(
            blur_directory=blur_directory,
            sharp_directory=sharp_directory,
            image_extension=image_extension,
            num_frames_per_blur=num_time_steps_per_frame,
            downscale_factor=image_downscale_factor,
        )

        self.model = Model(
            image_width=self.dataset.image_width,
            image_height=self.dataset.image_height,
            num_time_steps_per_frame=num_time_steps_per_frame,
            num_frames=len(self.dataset),
            num_coarse_samples_per_ray=num_coarse_samples_per_ray,
            num_fine_samples_per_ray=num_fine_samples_per_ray,
            near_depth=near_depth,
            far_depth=far_depth,
            num_scene_trajectory_basis_coefficients=num_scene_trajectory_basis_coefficients,
            num_camera_trajectory_basis_coefficients=num_camera_trajectory_basis_coefficients,
            num_voxels_per_axis=num_voxels_per_axis,
            min_bound_per_axis=min_bound_per_axis,
            max_bound_per_axis=max_bound_per_axis,
            voxel_dim=voxel_dim,
            color_model_hidden_dim=color_model_hidden_dim,
            device=self.accelerator.device,
        )

        self.batch_size = batch_size

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.step = Step(0, self.batch_size)
        self.accelerator.register_for_checkpointing(self.step)
        self.train_yielder = self._yield_data(self.dataloader)

    def _save_checkpoint(self, name: str):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self.results_directory / name)

    def load_checkpoint(self, name: str):
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(self.results_directory / name)

    @staticmethod
    def _yield_data(dataloader, skipped_dataloader=None) -> Dict[str, Tensor]:
        if exists(skipped_dataloader):
            for data in skipped_dataloader:
                yield data
        while True:
            for data in dataloader:
                yield data

    @staticmethod
    def _calculate_losses(output, sample):
        loss = blur_combination_loss(
            rearrange(
                output,
                "b t c h w -> (b h w) t c",
            ),
            rearrange(
                sample["blur_image"],
                "b h w c -> (b h w) c",
            ),
        )
        return loss

    def train(self):
        with tqdm(
            initial=self.step.step,
            total=self.num_steps,
            disable=not self.accelerator.is_main_process,
        ) as progress_bar:
            while self.step.step < self.num_steps:
                sample = next(self.train_yielder)
                output = self.model(sample["time_step"].to(self.accelerator.device), warmup=False)

                loss = self._calculate_losses(output.to(self.accelerator.device), sample.to(self.accelerator.device))

                self.accelerator.backward(loss)

                progress_bar.set_description(f"Loss: {loss.item():.4f} ")

                self.accelerator.wait_for_everyone()

                self.optimizer.step()

                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()

                if exists(self.inject_function):
                    self.inject_function(self.step, loss)

                self.step.step += 1

                progress_bar.update(1)

                # Checkpointing
                if self.step % self.num_steps_per_save == 0:
                    self._save_checkpoint(f"checkpoint_{self.step}")
