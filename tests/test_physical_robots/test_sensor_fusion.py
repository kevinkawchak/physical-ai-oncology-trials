"""Tests for examples-new/02_sensor_fusion_intraoperative.py

Covers sensor types, data models, segmentation, and the fusion pipeline.
"""

from __future__ import annotations

import numpy as np


class TestEnums:
    def test_sensor_types(self, sensor_fusion_mod):
        st = sensor_fusion_mod.SensorType
        assert st.STEREO_CAMERA.value is not None
        assert st.FORCE_TORQUE.value is not None
        assert st.OPTICAL_TRACKER.value is not None


class TestDepthImage:
    def test_creation(self, sensor_fusion_mod):
        depth = np.random.rand(480, 640).astype(np.float32) * 0.3
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        di = sensor_fusion_mod.DepthImage(
            depth_m=depth,
            rgb=rgb,
            timestamp_ns=1_000_000_000,
        )
        assert di.depth_m.shape == (480, 640)
        assert di.timestamp_ns == 1_000_000_000


class TestPointCloud:
    def test_creation(self, sensor_fusion_mod):
        pts = np.random.rand(100, 3).astype(np.float32)
        pc = sensor_fusion_mod.PointCloud(points_m=pts)
        assert pc.points_m.shape == (100, 3)
        assert pc.frame_id == "camera"


class TestTrackedPose:
    def test_creation(self, sensor_fusion_mod):
        pose = sensor_fusion_mod.TrackedPose(
            position_m=np.array([0.1, 0.0, 0.2]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            tool_id="instrument_1",
        )
        assert pose.tool_id == "instrument_1"
        assert pose.is_visible is True


class TestIntraoperativeScene:
    def test_creation(self, sensor_fusion_mod):
        scene = sensor_fusion_mod.IntraoperativeScene(timestamp_ns=1000)
        assert scene.instrument_visible is False
        assert scene.tissue_deformation_m == 0.0

    def test_margin_safe(self, sensor_fusion_mod):
        scene = sensor_fusion_mod.IntraoperativeScene(
            timestamp_ns=1000,
            tumor_margin_distance_m=0.01,
        )
        assert scene.is_margin_safe(min_margin_m=0.005) is True

    def test_margin_unsafe(self, sensor_fusion_mod):
        scene = sensor_fusion_mod.IntraoperativeScene(
            timestamp_ns=1000,
            tumor_margin_distance_m=0.002,
        )
        assert scene.is_margin_safe(min_margin_m=0.005) is False


class TestInstrumentSegmenter:
    def test_creation(self, sensor_fusion_mod):
        seg = sensor_fusion_mod.InstrumentSegmenter()
        assert seg is not None

    def test_segment_returns_mask(self, sensor_fusion_mod):
        seg = sensor_fusion_mod.InstrumentSegmenter()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = seg.segment(image)
        assert mask.shape[:2] == (480, 640)


class TestDepthToPointCloud:
    def test_conversion(self, sensor_fusion_mod, identity_camera_matrix):
        converter = sensor_fusion_mod.DepthToPointCloud(
            camera_matrix=identity_camera_matrix,
        )
        depth = np.random.rand(10, 10).astype(np.float32) * 0.3
        rgb = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        di = sensor_fusion_mod.DepthImage(
            depth_m=depth,
            rgb=rgb,
            timestamp_ns=1000,
        )
        pc = converter.convert(di)
        assert isinstance(pc, sensor_fusion_mod.PointCloud)
        assert pc.points_m.shape[1] == 3


class TestTemporalSynchronizer:
    def test_creation(self, sensor_fusion_mod):
        sync = sensor_fusion_mod.TemporalSynchronizer()
        assert sync is not None

    def test_add_stream(self, sensor_fusion_mod):
        sync = sensor_fusion_mod.TemporalSynchronizer()
        sync.add_stream("depth", buffer_size=5)
        # No error means stream was added
