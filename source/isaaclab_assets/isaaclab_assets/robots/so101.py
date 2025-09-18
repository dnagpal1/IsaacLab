# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-101 ARM robot.

The following configurations are available:

* :obj:`SO101_CFG`: SO-101 ARM robot with 6 joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)
* :obj:`SO101_HIGH_PD_CFG`: SO-101 ARM robot with stiffer PD control for IK tracking

Reference: https://github.com/TheRobotStudio/SO-ARM100
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import os

##
# Configuration
##

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Local path to SO-101 robot USD file
        usd_path=os.path.join(os.path.dirname(__file__), "../../../isaaclab/data/Robots/SO101/so101_instanceable.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Based on your joint names from the ROS script
            "Rotation": 0.0,        # Base rotation
            "Pitch": -0.5,          # Shoulder pitch (slight forward lean)
            "Elbow": 0.0,           # Elbow neutral
            "Wrist_Pitch": -1.5,    # Wrist pitch (downward)
            "Wrist_Roll": 0.0,      # Wrist roll neutral
            "Jaw": 0.02,            # Gripper slightly open
        },
    ),
    actuators={
        # Joint actuators based on your 6 joints
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            effort_limit=100.0,
            velocity_limit=2.0,
            stiffness=40.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit=50.0,
            velocity_limit=1.0,
            stiffness=20.0,
            damping=5.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

SO101_HIGH_PD_CFG = SO101_CFG.copy()
SO101_HIGH_PD_CFG.actuators["arm"].stiffness = 80.0
SO101_HIGH_PD_CFG.actuators["arm"].damping = 20.0
SO101_HIGH_PD_CFG.actuators["gripper"].stiffness = 40.0
SO101_HIGH_PD_CFG.actuators["gripper"].damping = 10.0