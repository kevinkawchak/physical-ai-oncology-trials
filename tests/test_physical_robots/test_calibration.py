"""Unit tests for examples-new/04_hand_eye_calibration_registration.py.

Tests cover transform utilities, HandEyeCalibrator,
PatientRegistration, and CalibrationVerifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/04_hand_eye_calibration_registration.py"


@pytest.fixture()
def mod():
    return load_module("hand_eye_calibration", MOD_PATH)


class TestTransformUtilities:
    def test_make_transform_identity(self, mod):
        T = mod.make_transform(np.eye(3), np.zeros(3))
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T, np.eye(4))

    def test_invert_transform(self, mod):
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = mod.make_transform(R, t)
        T_inv = mod.invert_transform(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_points(self, mod):
        T = mod.make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        transformed = mod.transform_points(T, pts)
        np.testing.assert_allclose(transformed[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(transformed[1], [2.0, 0.0, 0.0])

    def test_rotation_error(self, mod):
        T1 = mod.make_transform(np.eye(3), np.zeros(3))
        T2 = mod.make_transform(np.eye(3), np.zeros(3))
        err = mod.rotation_error_deg(T1, T2)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_translation_error(self, mod):
        T1 = mod.make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
        T2 = mod.make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
        err = mod.translation_error_mm(T1, T2)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_translation_error_nonzero(self, mod):
        T1 = mod.make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
        T2 = mod.make_transform(np.eye(3), np.array([3.0, 4.0, 0.0]))
        err = mod.translation_error_mm(T1, T2)
        # translation_error_mm returns norm * 1000 (meters -> mm)
        assert err == pytest.approx(5000.0, rel=0.01)


class TestHandEyeCalibrator:
    def test_instantiation(self, mod):
        cal = mod.HandEyeCalibrator()
        assert cal is not None

    def test_synthetic_calibration(self, mod):
        from scipy.spatial.transform import Rotation

        cal = mod.HandEyeCalibrator(method="eye_in_hand")
        # Build synthetic calibration poses (same approach as run_calibration_demo)
        true_T = mod.make_transform(
            Rotation.from_euler("xyz", [5, -3, 2], degrees=True).as_matrix(),
            np.array([0.02, -0.01, 0.05]),
        )
        rng = np.random.RandomState(42)
        for _ in range(20):
            R_ee = Rotation.from_euler("xyz", rng.uniform(-30, 30, 3), degrees=True).as_matrix()
            t_ee = rng.uniform(-0.1, 0.1, 3)
            T_base_to_ee = mod.make_transform(R_ee, t_ee)
            T_target_fixed = mod.make_transform(np.eye(3), np.array([0.3, 0.0, 0.0]))
            T_cam = mod.invert_transform(true_T) @ mod.invert_transform(T_base_to_ee) @ T_target_fixed
            T_cam[:3, 3] += rng.randn(3) * 0.0002
            cal.add_pose(mod.CalibrationPose(T_base_to_ee=T_base_to_ee, T_camera_to_target=T_cam))
        result = cal.calibrate()
        assert result.n_poses == 20
        assert result.T_hand_eye.shape == (4, 4)

    def test_too_few_poses_raises(self, mod):
        cal = mod.HandEyeCalibrator()
        cal.add_pose(mod.CalibrationPose(T_base_to_ee=np.eye(4), T_camera_to_target=np.eye(4)))
        cal.add_pose(mod.CalibrationPose(T_base_to_ee=np.eye(4), T_camera_to_target=np.eye(4)))
        with pytest.raises((ValueError, RuntimeError)):
            cal.calibrate()


class TestPatientRegistration:
    def test_identity_registration(self, mod):
        reg = mod.PatientRegistration()
        pts_src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pts_dst = pts_src.copy()
        for i in range(len(pts_src)):
            reg.add_fiducial(f"fid_{i}", pts_src[i], pts_dst[i])
        result = reg.register_fiducial()
        np.testing.assert_allclose(result.T_imaging_to_robot, np.eye(4), atol=1e-8)

    def test_known_translation(self, mod):
        reg = mod.PatientRegistration()
        pts_src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        offset = np.array([5.0, 3.0, 1.0])
        pts_dst = pts_src + offset
        for i in range(len(pts_src)):
            reg.add_fiducial(f"fid_{i}", pts_src[i], pts_dst[i])
        result = reg.register_fiducial()
        np.testing.assert_allclose(result.T_imaging_to_robot[:3, 3], offset, atol=1e-6)

    def test_too_few_points_raises(self, mod):
        reg = mod.PatientRegistration()
        reg.add_fiducial("fid_0", np.array([0, 0, 0], dtype=float), np.array([0, 0, 0], dtype=float))
        reg.add_fiducial("fid_1", np.array([1, 0, 0], dtype=float), np.array([1, 0, 0], dtype=float))
        with pytest.raises((ValueError, RuntimeError)):
            reg.register_fiducial()


class TestCalibrationVerifier:
    def test_verify_identity(self, mod):
        # Identity hand-eye and identity registration
        verifier = mod.CalibrationVerifier(
            T_hand_eye=np.eye(4),
            T_imaging_to_robot=np.eye(4),
        )
        # Add a test point where imaging == physical (identity transform)
        verifier.add_test_point(
            "test_origin",
            imaging_position_m=np.array([0.1, 0.2, 0.3]),
            physical_position_m=np.array([0.1, 0.2, 0.3]),
        )
        report = verifier.verify()
        assert report["acceptable"] is True
        assert report["mean_error_mm"] == pytest.approx(0.0, abs=1e-6)
        assert report["max_error_mm"] == pytest.approx(0.0, abs=1e-6)
