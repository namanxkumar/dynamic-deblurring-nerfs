import torch
from model.model import Model

device = torch.device("cpu")

model = Model(
    image_width=640,
    image_height=352,
    num_time_steps_per_frame=7,
    num_frames=76,
    num_coarse_samples_per_ray=4,
    num_fine_samples_per_ray=4,
    near_depth=0.1,
    far_depth=10.0,
    num_scene_trajectory_basis_coefficients=4,
    num_camera_trajectory_basis_coefficients=4,
    num_voxels_per_axis=8,
    min_bound_per_axis=1.0,
    max_bound_per_axis=10.0,
    voxel_dim=4,
    color_model_hidden_dim=4,
    device=device,
)

sample_output = model(
    torch.tensor([3])
)

print(sample_output)