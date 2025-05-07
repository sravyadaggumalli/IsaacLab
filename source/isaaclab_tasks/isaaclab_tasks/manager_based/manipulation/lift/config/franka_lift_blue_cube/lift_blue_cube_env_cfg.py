# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This file defines the environment configuration for the custom
# Isaac-Lift-BlueCube-Franka-v0 task.
# It inherits from the base LiftEnvCfg and overrides MDP managers
# (observations, rewards, terminations, events) using specialized
# config classes defined below to handle the 3-cube scenario.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs from assets and markers
##
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


##
# Specialized MDP Configurations for the Blue Cube Lifting & Placing Task
##


@configclass
class BlueCubeLiftPlaceCommandsCfg:
    """Command terms for the MDP - Target pose on the table near robot."""

    placement_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set in __post_init__
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Adjusted ranges to place the blue cube closer to the robot
            pos_x=(0.25, 0.4), pos_y=(-0.2, 0.2), pos_z=(0.02, 0.02), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class BlueCubeLiftPlaceActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class BlueCubeLiftPlaceObservationsCfg:
    """Observation specifications - Modified for 3 cubes."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)
        blue_cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube_blue")})
        red_cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube_red")})
        green_cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube_green")})
        target_placement_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "placement_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class BlueCubeLiftPlaceEventCfg:
    """Configuration for events."""

    reset_base = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_blue_cube_position = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (0.25, 0.35), "y": (-0.05, 0.05), "z": (0.02, 0.025)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cube_blue")})
    reset_red_cube_position = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (0.2, 0.3), "y": (-0.15, 0.05), "z": (0.02, 0.025)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cube_red")})
    reset_green_cube_position = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (0.2, 0.3), "y": (0.05, 0.15), "z": (0.02, 0.025)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cube_green")})


@configclass
class BlueCubeLiftPlaceRewardsCfg:
    """Reward terms for the MDP."""

    # Initial reach has decent reward
    reaching_blue_cube = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_blue")},
        weight=5.0 # prior value: 1.0
    )

    # reward lifting the blue cube higher to encourage exploration
    lifting_blue_cube = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_blue")},
        weight=10.0 # prior value: 15.0
    )

    # reward holding higher than lifting to encourage holding otherwise the agent is gaming to do only reach and lift
    holding_blue_cube = RewTerm(
        func=mdp.reward_holding_object,
        params={
            "minimal_height": 0.04,
            "std_distance": 0.2,
            "command_name": "placement_pose",
            "object_cfg": SceneEntityCfg("cube_blue"),
            "robot_cfg": SceneEntityCfg("robot"),
            "closed_joint_pos_thresh": 0.01
        },
        weight=15.0 # new intermediate reward
    )

    # Higest Reward for placing the blue cube near the target on the table
    blue_cube_placement_reward = RewTerm(
        func=mdp.reward_cube_placement,
        params={
            "std": 0.1, # Std dev for distance reward scaling
            "table_height_threshold": 0.03,
            "command_name": "placement_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("cube_blue")
        },
        weight=25.0 # Make placement highly rewarding
    )

    # Reduce action penalties to avoid discouraging initial exploration
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-6 # prior value: -1e-4
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-6, # prior value -1e-4
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Keep penalties for interacting with wrong cubes
    grasping_red_cube_penalty = RewTerm(func=mdp.penalize_wrong_grasp, params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame"), "distractor_cube_cfg": SceneEntityCfg("cube_red")}, weight=-5.0)
    grasping_green_cube_penalty = RewTerm(func=mdp.penalize_wrong_grasp, params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame"), "distractor_cube_cfg": SceneEntityCfg("cube_green")}, weight=-5.0)


@configclass
class BlueCubeLiftPlaceTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    blue_cube_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_blue")})


@configclass
class BlueCubeLiftPlaceCurriculumCfg:
    """Curriculum terms - Inherited from base lift (can be modified)."""
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000})


##
# Environment configuration
##

@configclass
class FrankaBlueCubeLiftEnvCfg(LiftEnvCfg):
    """Configuration for the lifting and placing the blue cube environment."""


    # Basic settings
    observations: BlueCubeLiftPlaceObservationsCfg = BlueCubeLiftPlaceObservationsCfg()
    actions: BlueCubeLiftPlaceActionsCfg = BlueCubeLiftPlaceActionsCfg()
    commands: BlueCubeLiftPlaceCommandsCfg = BlueCubeLiftPlaceCommandsCfg()
    # MDP settings
    rewards: BlueCubeLiftPlaceRewardsCfg = BlueCubeLiftPlaceRewardsCfg()
    terminations: BlueCubeLiftPlaceTerminationsCfg = BlueCubeLiftPlaceTerminationsCfg()
    events: BlueCubeLiftPlaceEventCfg = BlueCubeLiftPlaceEventCfg()
    curriculum: BlueCubeLiftPlaceCurriculumCfg = BlueCubeLiftPlaceCurriculumCfg()


    def __post_init__(self):
        """Post-initialization."""
        # post init of parent
        super().__post_init__() 

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(asset_name="robot", joint_names=["panda_finger.*"], open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0})

        # Set Franka End-Effector as body name of 'placement_pose' command
        self.commands.placement_pose.body_name = "panda_hand"

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_angular_velocity=1000.0, max_linear_velocity=1000.0, max_depenetration_velocity=5.0, disable_gravity=False)
        init_z = 0.0203

        # Set each stacking cube deterministically
        self.scene.cube_blue = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube_Blue", init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, init_z], rot=[1, 0, 0, 0]), spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=cube_properties, semantic_tags=[("class", "target_cube"), ("class", "cube_blue")]))
        self.scene.cube_red = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube_Red", init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.1, init_z], rot=[1, 0, 0, 0]), spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=cube_properties, semantic_tags=[("class", "distractor_cube"), ("class", "cube_red")]))
        self.scene.cube_green = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube_Green", init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.1, init_z], rot=[1, 0, 0, 0]), spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=cube_properties, semantic_tags=[("class", "distractor_cube"), ("class", "cube_green")]))
        self.scene.object = self.scene.cube_blue

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0", debug_vis=False, visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector", offset=OffsetCfg(pos=[0.0, 0.0, 0.1034])),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", name="tool_rightfinger", offset=OffsetCfg(pos=(0.0, 0.0, 0.046))),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", name="tool_leftfinger", offset=OffsetCfg(pos=(0.0, 0.0, 0.046))),
            ]
        )