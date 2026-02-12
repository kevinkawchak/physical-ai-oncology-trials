from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("sensor_fusion_intraoperative", "examples-new/02_sensor_fusion_intraoperative.py")


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for SensorType enum."""

    def test_sensor_type_stereo_camera(self, mod):
        assert mod.SensorType.STEREO_CAMERA is not None

    def test_sensor_type_depth_camera(self, mod):
        assert mod.SensorType.DEPTH_CAMERA is not None

    def test_sensor_type_force_torque(self, mod):
        assert mod.SensorType.FORCE_TORQUE is not None

    def test_sensor_type_joint_encoder(self, mod):
        assert mod.SensorType.JOINT_ENCODER is not None

    def test_sensor_type_optical_tracker(self, mod):
        assert mod.SensorType.OPTICAL_TRACKER is not None

    def test_sensor_type_endoscope(self, mod):
        assert mod.SensorType.ENDOSCOPE is not None


# ---------------------------------------------------------------------------
# TestStereoImagePair
# ---------------------------------------------------------------------------


class TestStereoImagePair:
    """Tests for StereoImagePair dataclass."""

    def test_creation(self, mod):
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        pair = mod.StereoImagePair(left=left, right=right, timestamp_ns=1000)
        assert pair.timestamp_ns == 1000

    def test_default_baseline(self, mod):
        left = np.zeros((2, 2, 3), dtype=np.uint8)
        right = np.zeros((2, 2, 3), dtype=np.uint8)
        pair = mod.StereoImagePair(left=left, right=right, timestamp_ns=0)
        assert pair.baseline_m == 0.005

    def test_default_camera_matrices_identity(self, mod):
        left = np.zeros((2, 2, 3), dtype=np.uint8)
        right = np.zeros((2, 2, 3), dtype=np.uint8)
        pair = mod.StereoImagePair(left=left, right=right, timestamp_ns=0)
        np.testing.assert_array_equal(pair.camera_matrix_left, np.eye(3))


# ---------------------------------------------------------------------------
# TestDepthImage
# ---------------------------------------------------------------------------


class TestDepthImage:
    """Tests for DepthImage dataclass."""

    def test_creation(self, mod):
        depth = np.full((480, 640), 0.2, dtype=np.float32)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        di = mod.DepthImage(depth_m=depth, rgb=rgb, timestamp_ns=100)
        assert di.timestamp_ns == 100

    def test_depth_shape(self, mod):
        depth = np.full((100, 200), 0.1, dtype=np.float32)
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        di = mod.DepthImage(depth_m=depth, rgb=rgb, timestamp_ns=0)
        assert di.depth_m.shape == (100, 200)


# ---------------------------------------------------------------------------
# TestPointCloud
# ---------------------------------------------------------------------------


class TestPointCloud:
    """Tests for PointCloud dataclass."""

    def test_creation(self, mod):
        points = np.random.randn(100, 3).astype(np.float32)
        pc = mod.PointCloud(points_m=points)
        assert pc.points_m.shape == (100, 3)

    def test_default_frame_id(self, mod):
        points = np.zeros((1, 3), dtype=np.float32)
        pc = mod.PointCloud(points_m=points)
        assert pc.frame_id == "camera"

    def test_colors_none_by_default(self, mod):
        points = np.zeros((1, 3), dtype=np.float32)
        pc = mod.PointCloud(points_m=points)
        assert pc.colors is None


# ---------------------------------------------------------------------------
# TestTrackedPose
# ---------------------------------------------------------------------------


class TestTrackedPose:
    """Tests for TrackedPose dataclass."""

    def test_creation(self, mod):
        pos = np.array([0.1, 0.2, 0.3])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tp = mod.TrackedPose(position_m=pos, orientation_quat=quat, tool_id="probe")
        assert tp.tool_id == "probe"

    def test_default_visible(self, mod):
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tp = mod.TrackedPose(position_m=pos, orientation_quat=quat, tool_id="t")
        assert tp.is_visible is True

    def test_timestamp_default(self, mod):
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tp = mod.TrackedPose(position_m=pos, orientation_quat=quat, tool_id="t")
        assert tp.timestamp_ns == 0


# ---------------------------------------------------------------------------
# TestInstrumentSegmenter
# ---------------------------------------------------------------------------


class TestInstrumentSegmenter:
    """Tests for InstrumentSegmenter class."""

    def test_init(self, mod):
        seg = mod.InstrumentSegmenter()
        assert seg.input_size == (480, 640)

    def test_segment_returns_mask(self, mod):
        seg = mod.InstrumentSegmenter()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = seg.segment(img)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_segment_has_tip_class(self, mod):
        seg = mod.InstrumentSegmenter()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = seg.segment(img)
        assert np.any(mask == seg.CLASS_TIP)

    def test_get_tip_position(self, mod):
        seg = mod.InstrumentSegmenter()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = seg.segment(img)
        tip = seg.get_tip_position_pixels(mask)
        assert tip is not None
        assert tip.shape == (2,)


# ---------------------------------------------------------------------------
# TestDepthToPointCloud
# ---------------------------------------------------------------------------


class TestDepthToPointCloud:
    """Tests for DepthToPointCloud class."""

    def test_init(self, mod):
        cam = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        converter = mod.DepthToPointCloud(cam)
        assert converter.max_depth_m == 0.5

    def test_convert_produces_points(self, mod):
        cam = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        converter = mod.DepthToPointCloud(cam, max_depth_m=1.0)
        depth = np.full((48, 64), 0.2, dtype=np.float32)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        di = mod.DepthImage(depth_m=depth, rgb=rgb, timestamp_ns=0)
        pc = converter.convert(di)
        assert pc.points_m.shape[1] == 3
        assert pc.points_m.shape[0] > 0


# ---------------------------------------------------------------------------
# TestTemporalSynchronizer
# ---------------------------------------------------------------------------


class TestTemporalSynchronizer:
    """Tests for TemporalSynchronizer class."""

    def test_init(self, mod):
        sync = mod.TemporalSynchronizer(tolerance_ns=5_000_000)
        assert sync.tolerance_ns == 5_000_000

    def test_add_stream(self, mod):
        sync = mod.TemporalSynchronizer()
        sync.add_stream("camera", buffer_size=5)
        assert "camera" in sync._buffers

    def test_push_and_get_none_when_incomplete(self, mod):
        sync = mod.TemporalSynchronizer()
        sync.add_stream("a")
        sync.add_stream("b")
        sync.push("a", 1000, "data_a")
        result = sync.get_synchronized()
        assert result is None

    def test_get_synchronized_returns_bundle(self, mod):
        sync = mod.TemporalSynchronizer(tolerance_ns=10_000_000)
        sync.add_stream("a")
        sync.add_stream("b")
        sync.push("a", 1000, "data_a")
        sync.push("b", 1000, "data_b")
        result = sync.get_synchronized()
        assert result is not None
        assert "a" in result
        assert "b" in result


# ---------------------------------------------------------------------------
# TestSensorFusionPipeline
# ---------------------------------------------------------------------------


class TestSensorFusionPipeline:
    """Tests for SensorFusionPipeline class."""

    def test_init(self, mod):
        cam = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        pipeline = mod.SensorFusionPipeline(camera_matrix=cam)
        assert pipeline.camera_matrix is cam

    def test_process_returns_scene(self, mod):
        cam = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        pipeline = mod.SensorFusionPipeline(camera_matrix=cam)
        depth = np.full((480, 640), 0.2, dtype=np.float32)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        di = mod.DepthImage(depth_m=depth, rgb=rgb, timestamp_ns=0, camera_matrix=cam)
        scene = pipeline.process(
            depth_image=di,
            force_xyz_n=np.zeros(3),
            instrument_ee_position_m=np.array([0.0, 0.0, -0.15]),
            timestamp_ns=0,
        )
        assert scene.timestamp_ns == 0
        assert scene.tissue_point_cloud is not None
