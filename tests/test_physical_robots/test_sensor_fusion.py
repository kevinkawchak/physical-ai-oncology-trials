"""Unit tests for examples-new/02_sensor_fusion_intraoperative.py.

Tests cover SensorType enum and associated dataclasses:
StereoImagePair, DepthImage, PointCloud, TrackedPose.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/02_sensor_fusion_intraoperative.py"


@pytest.fixture()
def mod():
    return load_module("sensor_fusion", MOD_PATH)


class TestEnums:
    def test_sensor_type(self, mod):
        names = [e.name for e in mod.SensorType]
        for s in ("STEREO_CAMERA", "DEPTH_CAMERA", "FORCE_TORQUE", "OPTICAL_TRACKER"):
            assert s in names


class TestStereoImagePair:
    def test_creation(self, mod):
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        pair = mod.StereoImagePair(
            left=left,
            right=right,
            timestamp_ns=1_000_000_000,
            camera_matrix_left=np.eye(3),
            camera_matrix_right=np.eye(3),
            baseline_m=0.06,
        )
        assert pair.baseline_m == 0.06
        assert pair.left.shape == (480, 640, 3)


class TestDepthImage:
    def test_creation(self, mod):
        depth = np.ones((480, 640), dtype=np.float32) * 0.5
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        di = mod.DepthImage(
            depth_m=depth,
            rgb=rgb,
            timestamp_ns=1_000_000_000,
            camera_matrix=np.eye(3),
        )
        assert di.depth_m.shape == (480, 640)
        assert np.mean(di.depth_m) == pytest.approx(0.5)


class TestPointCloud:
    def test_creation(self, mod):
        pts = np.random.randn(1000, 3).astype(np.float32)
        pc = mod.PointCloud(
            points_m=pts,
            frame_id="camera_link",
            timestamp_ns=1_000_000_000,
        )
        assert pc.points_m.shape == (1000, 3)
        assert pc.frame_id == "camera_link"

    def test_empty_cloud(self, mod):
        pc = mod.PointCloud(
            points_m=np.empty((0, 3), dtype=np.float32),
            frame_id="empty",
            timestamp_ns=0,
        )
        assert pc.points_m.shape[0] == 0


class TestTrackedPose:
    def test_creation(self, mod):
        pose = mod.TrackedPose(
            position_m=np.array([0.1, 0.2, 0.3]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            tool_id="scalpel",
            is_visible=True,
            rms_error_mm=0.5,
            timestamp_ns=1_000_000_000,
        )
        assert pose.tool_id == "scalpel"
        assert pose.is_visible is True
        assert pose.rms_error_mm < 1.0

    def test_invisible_tool(self, mod):
        pose = mod.TrackedPose(
            position_m=np.zeros(3),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            tool_id="occluded_tool",
            is_visible=False,
            rms_error_mm=float("inf"),
            timestamp_ns=1_000_000_000,
        )
        assert pose.is_visible is False
