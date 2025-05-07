# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    table_z = env.scene.env_origins[:, 2] # Get the Z-origin for each env
    return torch.where(object.data.root_pos_w[:, 2] > table_z + minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Use hardcoded index for 'end_effector'
    ee_idx = 0
    num_targets = ee_frame.data.target_pos_w.shape[-2]
    if not (ee_idx < num_targets):
         print(f"Warning: 'end_effector' frame index ({ee_idx}) out of bounds for FrameTransformer data shape {ee_frame.data.target_pos_w.shape}. Check FrameTransformerCfg.")
         return torch.zeros(env.num_envs, device=env.device)

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., ee_idx, :] # Use the index
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[robot_cfg.name] # Robot is Articulation
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3] # Target pose is relative to robot base
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    table_z = env.scene.env_origins[:, 2]
    object_is_lifted_flag = object.data.root_pos_w[:, 2] > table_z + minimal_height
    # rewarded if the object is lifted above the threshold
    return object_is_lifted_flag * (1 - torch.tanh(distance / std))


def penalize_wrong_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distractor_cube_cfg: SceneEntityCfg = SceneEntityCfg("object"), # Provide the specific distractor cube
    close_grasp_dist_thresh: float = 0.05, # Threshold for finger distance to cube center
    closed_joint_pos_thresh: float = 0.01, # Threshold for sum of finger joint positions to consider closed
) -> torch.Tensor:
    
    """Penalize the agent for grasping a distractor cube."""
    robot: Articulation = env.scene[robot_cfg.name]
    distractor_cube: RigidObject = env.scene[distractor_cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Check if gripper is closed based on joint positions
    finger_joint_ids = robot.find_joints("panda_finger_joint.*")[0]
    if len(finger_joint_ids) != 2:
        return torch.zeros(env.num_envs, device=env.device)
    finger_joint_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper_is_closed = finger_joint_pos.sum(dim=1) < closed_joint_pos_thresh

    # Check proximity of fingertips to the distractor cube
    # Use hardcoded indices based on FrameTransformerCfg order end_effector (0), tool_rightfinger (1), tool_leftfinger (2)
    right_finger_idx = 1
    left_finger_idx = 2

    # Safety check for indices
    num_targets = ee_frame.data.target_pos_w.shape[-2]
    if not (right_finger_idx < num_targets and left_finger_idx < num_targets):
         print(f"Warning: Fingertip frame indices ({left_finger_idx}, {right_finger_idx}) out of bounds for FrameTransformer data shape {ee_frame.data.target_pos_w.shape}. Check FrameTransformerCfg.")
         return torch.zeros(env.num_envs, device=env.device)

    # Access data using the correct indices
    left_finger_pos_w = ee_frame.data.target_pos_w[..., left_finger_idx, :]
    right_finger_pos_w = ee_frame.data.target_pos_w[..., right_finger_idx, :]

    distractor_pos_w = distractor_cube.data.root_pos_w[:, :3]
    dist_left_finger = torch.norm(left_finger_pos_w - distractor_pos_w, dim=1)
    dist_right_finger = torch.norm(right_finger_pos_w - distractor_pos_w, dim=1)
    fingers_are_close = (dist_left_finger < close_grasp_dist_thresh) & (dist_right_finger < close_grasp_dist_thresh)

    # Apply penalty if gripper is closed AND fingers are close to the distractor
    penalty = (gripper_is_closed & fingers_are_close).float()
    return penalty

def reward_cube_placement(
    env: ManagerBasedRLEnv,
    std: float,
    table_height_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for placing the object near the target pose ON THE TABLE."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Target pose from command (relative to robot base)
    des_pose_b = command[:, :7]
    # Convert target pose to world frame
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pose_b[:, :3], des_pose_b[:, 3:7])

    # Current object pose
    object_pos_w = object.data.root_pos_w[:, :3]
    object_z = object_pos_w[:, 2]
    table_z = env.scene.env_origins[:, 2]
    object_is_on_table = object_z < table_z + table_height_threshold

    # Calculate distance to target
    distance = torch.norm(des_pos_w[:, :2] - object_pos_w[:, :2], dim=1)

    # rewarded if the object is on the table and distance above the threshold
    placement_reward = object_is_on_table * (1 - torch.tanh(distance / std))

    return placement_reward


def reward_holding_object(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    std_distance: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    closed_joint_pos_thresh: float = 0.01,
) -> torch.Tensor:
    
    """Reward the agent for holding the object above a height AND moving towards the target position."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Check if Object is lifted
    table_z = env.scene.env_origins[:, 2]
    object_is_lifted_flag = object.data.root_pos_w[:, 2] > table_z + minimal_height

    # Check if Gripper is closed
    finger_joint_ids = robot.find_joints("panda_finger_joint.*")[0]
    if len(finger_joint_ids) != 2:
        # Return zero reward if joints aren't found correctly
        return torch.zeros_like(object_is_lifted_flag).float()
    finger_joint_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper_is_closed = finger_joint_pos.sum(dim=1) < closed_joint_pos_thresh

    is_holding = (object_is_lifted_flag & gripper_is_closed)

    # Get Distance to target placement pose
    command = env.command_manager.get_command(command_name)
    des_pose_b = command[:, :7] # Get target pose relative to robot base
    # Convert target pose to world frame
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pose_b[:, :3], des_pose_b[:, 3:7])
    # Current object position
    object_pos_w = object.data.root_pos_w[:, :3]
    # Calculate distance
    distance_to_target_pos = torch.norm(des_pos_w[:, :2] - object_pos_w[:, :2], dim=1)

    # Calculate reward based on distance
    distance_reward = (1 - torch.tanh(distance_to_target_pos / std_distance))

    # reward ONLY when actively holding the object
    final_reward = is_holding.float() * distance_reward

    return final_reward