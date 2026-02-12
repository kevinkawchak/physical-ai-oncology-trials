"""Fixtures for physical robot tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def sensor_fusion_mod():
    """Load the sensor fusion intraoperative module."""
    return load_module(
        "sensor_fusion",
        "examples-new/02_sensor_fusion_intraoperative.py",
    )


@pytest.fixture()
def ros2_mod():
    """Load the ROS2 surgical deployment module."""
    return load_module(
        "ros2_surgical",
        "examples-new/03_ros2_surgical_deployment.py",
    )


@pytest.fixture()
def shared_autonomy_mod():
    """Load the shared autonomy teleoperation module."""
    return load_module(
        "shared_autonomy",
        "examples-new/05_shared_autonomy_teleoperation.py",
    )


@pytest.fixture()
def identity_camera_matrix():
    """Identity camera matrix for testing."""
    return np.eye(3)
