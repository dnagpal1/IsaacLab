#!/usr/bin/env python3

"""Test script to check path resolution."""

import os

# Simulate the path resolution that happens in franka.py
franka_file_path = "/mnt/c/Users/dipen/Documents/IsaacLab-fork/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py"

# Get the directory of the franka.py file
franka_dir = os.path.dirname(franka_file_path)
print(f"Franka.py directory: {franka_dir}")

# Construct the path as done in franka.py
so101_path = os.path.join(franka_dir, "../../../isaaclab/data/Robots/SO101/so101_instanceable.usd")
print(f"Relative path: {so101_path}")

# Resolve the absolute path
abs_path = os.path.abspath(so101_path)
print(f"Absolute path: {abs_path}")

# Check if file exists
exists = os.path.exists(abs_path)
print(f"File exists: {exists}")

if exists:
    file_size = os.path.getsize(abs_path)
    print(f"File size: {file_size} bytes ({file_size / (1024*1024):.1f} MB)")
else:
    print("File does not exist!")

# Also check the actual location where we put the file
actual_path = "/mnt/c/Users/dipen/Documents/IsaacLab-fork/IsaacLab/source/isaaclab/data/Robots/SO101/so101_instanceable.usd"
actual_exists = os.path.exists(actual_path)
print(f"Actual file location exists: {actual_exists}")

if actual_exists:
    actual_size = os.path.getsize(actual_path)
    print(f"Actual file size: {actual_size} bytes ({actual_size / (1024*1024):.1f} MB)")