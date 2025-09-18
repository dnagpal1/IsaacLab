#!/usr/bin/env python3

"""Test script to check SO-101 configuration import and registration."""

import sys
import os

# Add IsaacLab to Python path
sys.path.append("/mnt/c/Users/dipen/Documents/IsaacLab-fork/IsaacLab/source")

try:
    print("Testing SO-101 configuration import...")

    # Try importing the SO-101 robot configuration
    print("1. Importing SO-101 robot config...")
    from isaaclab_assets.robots.so101 import SO101_CFG, SO101_HIGH_PD_CFG
    print("   ✓ SO-101 robot config imported successfully")

    # Try importing the SO-101 stack events
    print("2. Importing SO-101 stack events...")
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import so101_stack_events
    print("   ✓ SO-101 stack events imported successfully")

    # Try importing the SO-101 stack configurations
    print("3. Importing SO-101 stack configurations...")
    from isaaclab_tasks.manager_based.manipulation.stack.config.so101.stack_joint_pos_env_cfg import SO101CubeStackEnvCfg as SO101JointCfg
    from isaaclab_tasks.manager_based.manipulation.stack.config.so101.stack_ik_rel_env_cfg import SO101CubeStackEnvCfg as SO101IKCfg
    print("   ✓ SO-101 stack configurations imported successfully")

    # Try importing isaaclab_tasks (this should trigger auto-registration)
    print("4. Importing isaaclab_tasks (triggering auto-registration)...")
    import isaaclab_tasks
    print("   ✓ isaaclab_tasks imported successfully")

    # Check if environments are registered
    print("5. Checking gym environment registration...")
    import gymnasium as gym

    all_envs = list(gym.envs.registry.env_specs.keys())
    so101_envs = [env_id for env_id in all_envs if 'SO101' in env_id]
    stack_envs = [env_id for env_id in all_envs if 'Stack' in env_id]

    print(f"   Total environments: {len(all_envs)}")
    print(f"   Stack environments: {len(stack_envs)}")
    print(f"   SO-101 environments: {len(so101_envs)}")

    if so101_envs:
        print("   ✓ SO-101 environments found:")
        for env_id in so101_envs:
            print(f"     - {env_id}")
    else:
        print("   ✗ No SO-101 environments found")
        print("   Available Stack environments:")
        for env_id in stack_envs[:5]:  # Show first 5
            print(f"     - {env_id}")
        if len(stack_envs) > 5:
            print(f"     ... and {len(stack_envs) - 5} more")

    print("✓ Import test completed successfully!")

except Exception as e:
    print(f"✗ Import test failed: {e}")
    import traceback
    traceback.print_exc()