# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for SO-101 ARM robot stack environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(env: ManagerBasedEnv, env_ids: torch.Tensor, default_pose: list[float]):
    """Set the default joint pose for SO-101 ARM.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        default_pose: Default joint positions [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw].
    """
    # Extract the used robot
    robot: Articulation = env.scene["robot"]

    # Set joint positions based on your SO-101 joint configuration
    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    # Map the default pose to joint indices (adjust indices based on your URDF joint order)
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

    for i, joint_name in enumerate(joint_names):
        if i < len(default_pose):
            # Find joint index in the robot (you may need to adjust this based on your robot's configuration)
            if i < joint_pos.shape[1]:
                joint_pos[env_ids, i] = default_pose[i]

    # Set joint velocities to zero
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    # Apply the joint states
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize SO-101 ARM joint positions by adding Gaussian noise.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.
        asset_cfg: Configuration for the asset to randomize.
    """
    # Extract the used robot
    robot: Articulation = env.scene[asset_cfg.name]

    # Get current joint positions
    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    # Add Gaussian noise to joint positions (for arm joints only, excluding gripper)
    num_arm_joints = 5  # SO-101 has 5 arm joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
    noise = torch.normal(mean, std, size=(len(env_ids), num_arm_joints), device=env.device)
    joint_pos[env_ids, :num_arm_joints] += noise

    # Clamp joint positions to valid ranges (adjust limits based on your robot's specifications)
    joint_limits_low = torch.tensor([-3.14, -1.57, -2.09, -1.57, -3.14], device=env.device)
    joint_limits_high = torch.tensor([3.14, 1.57, 2.09, 1.57, 3.14], device=env.device)

    joint_pos[env_ids, :num_arm_joints] = torch.clamp(
        joint_pos[env_ids, :num_arm_joints],
        joint_limits_low,
        joint_limits_high
    )

    # Set joint velocities to zero
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    # Apply the joint states
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    min_separation: float,
    asset_cfgs: list[SceneEntityCfg],
):
    """Randomize object poses for cube stacking task.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        pose_range: Dictionary with position and orientation ranges.
        min_separation: Minimum distance between objects.
        asset_cfgs: List of asset configurations for the objects.
    """
    # Sample random poses for each object
    num_envs = len(env_ids)
    num_objects = len(asset_cfgs)

    # Extract position ranges
    x_range = pose_range["x"]
    y_range = pose_range["y"]
    z_range = pose_range["z"]
    yaw_range = pose_range.get("yaw", (0.0, 0.0))

    for obj_idx, asset_cfg in enumerate(asset_cfgs):
        # Get the object
        obj: Articulation = env.scene[asset_cfg.name]

        # Sample positions with minimum separation constraint
        valid_poses = False
        attempts = 0
        max_attempts = 100

        while not valid_poses and attempts < max_attempts:
            # Sample random positions
            x_pos = sample_uniform(x_range[0], x_range[1], (num_envs, 1), device=env.device)
            y_pos = sample_uniform(y_range[0], y_range[1], (num_envs, 1), device=env.device)
            z_pos = sample_uniform(z_range[0], z_range[1], (num_envs, 1), device=env.device)

            positions = torch.cat([x_pos, y_pos, z_pos], dim=1)

            # Check minimum separation from other objects
            valid_poses = True
            for other_idx, other_cfg in enumerate(asset_cfgs):
                if other_idx != obj_idx and other_idx < obj_idx:  # Only check against already placed objects
                    other_obj = env.scene[other_cfg.name]
                    other_pos = other_obj.data.root_pos_w[env_ids, :3]

                    # Calculate distances
                    distances = torch.norm(positions - other_pos, dim=1)
                    if torch.any(distances < min_separation):
                        valid_poses = False
                        break

            attempts += 1

        # Sample random orientations (only yaw rotation)
        yaw = sample_uniform(yaw_range[0], yaw_range[1], (num_envs, 1), device=env.device)

        # Convert yaw to quaternion (w, x, y, z)
        orientations = torch.zeros((num_envs, 4), device=env.device)
        orientations[:, 0] = torch.cos(yaw.squeeze() / 2)  # w
        orientations[:, 3] = torch.sin(yaw.squeeze() / 2)  # z

        # Combine position and orientation
        poses = torch.cat([positions, orientations], dim=1)

        # Set object pose
        obj.write_root_pose_to_sim(poses, env_ids=env_ids)