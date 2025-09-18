# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the object stack environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

# Import robot-specific configurations to ensure they're registered
from . import franka  # noqa: F401
from . import galbot  # noqa: F401
from . import so101  # noqa: F401
from . import ur10_gripper  # noqa: F401
