"""Tests for examples-new/04_hand_eye_calibration_registration.py

Covers rigid transform utilities, HandEyeCalibrator (Tsai-Lenz),
PatientRegistration (Arun SVD), and CalibrationVerifier.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


# ── Rigid transform utilities ─────────────────────────────────────────────


class TestTransformUtilities:
    def test_make_transform_identity(self, calibration_mod):
        T = calibration_mod.make_transform(np.eye(3), np.zeros(3))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_make_transform_translation_only(self, calibration_mod):
        T = calibration_mod.make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_invert_transform_identity(self, calibration_mod):
        T = np.eye(4)
        T_inv = calibration_mod.invert_transform(T)
        np.testing.assert_array_almost_equal(T_inv, np.eye(4))

    def test_invert_transform_roundtrip(self, calibration_mod):
        R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        t = np.array([0.1, -0.2, 0.3])
        T = calibration_mod.make_transform(R, t)
        T_inv = calibration_mod.invert_transform(T)
        product = T @ T_inv
        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)

    def test_transform_points_identity(self, calibration_mod):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = calibration_mod.transform_points(np.eye(4), pts)
        np.testing.assert_array_almost_equal(out, pts)

    def test_transform_points_translation(self, calibration_mod):
        T = calibration_mod.make_transform(np.eye(3), np.array([10.0, 0.0, 0.0]))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = calibration_mod.transform_points(T, pts)
        expected = np.array([[10.0, 0.0, 0.0], [11.0, 1.0, 1.0]])
        np.testing.assert_array_almost_equal(out, expected)

    def test_rotation_error_zero(self, calibration_mod):
        T = np.eye(4)
        err = calibration_mod.rotation_error_deg(T, T)
        assert abs(err) < 1e-10

    def test_rotation_error_90deg(self, calibration_mod):
        T1 = np.eye(4)
        R2 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        T2 = calibration_mod.make_transform(R2, np.zeros(3))
        err = calibration_mod.rotation_error_deg(T1, T2)
        assert abs(err - 90.0) < 0.1

    def test_translation_error_mm(self, calibration_mod):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [0.001, 0.0, 0.0]  # 1 mm in x
        err = calibration_mod.translation_error_mm(T1, T2)
        assert abs(err - 1.0) < 0.01


# ── HandEyeCalibrator ────────────────────────────────────────────────────


class TestHandEyeCalibrator:
    def _generate_synthetic_poses(self, calibration_mod, n_poses=10, T_hand_eye=None):
        """Generate synthetic calibration poses with known ground truth."""
        if T_hand_eye is None:
            R_he = Rotation.from_euler("xyz", [5, -3, 10], degrees=True).as_matrix()
            T_hand_eye = calibration_mod.make_transform(R_he, np.array([0.05, -0.02, 0.1]))

        T_target_world = calibration_mod.make_transform(
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix(),
            np.array([0.5, 0.0, 0.3]),
        )

        poses = []
        for i in range(n_poses):
            angles = np.random.uniform(-30, 30, 3)
            R_ee = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            t_ee = np.array([0.3 + i * 0.01, 0.1 - i * 0.005, 0.2])
            T_base_ee = calibration_mod.make_transform(R_ee, t_ee)

            T_cam_world = calibration_mod.invert_transform(T_hand_eye) @ calibration_mod.invert_transform(T_base_ee)
            T_cam_target = T_cam_world @ T_target_world

            pose = calibration_mod.CalibrationPose(
                T_base_to_ee=T_base_ee,
                T_camera_to_target=T_cam_target,
            )
            poses.append(pose)
        return poses

    def test_calibrate_minimum_poses(self, calibration_mod):
        calibrator = calibration_mod.HandEyeCalibrator()
        poses = self._generate_synthetic_poses(calibration_mod, n_poses=5)
        for p in poses:
            calibrator.add_pose(p)
        result = calibrator.calibrate()
        assert result.T_hand_eye.shape == (4, 4)

    def test_calibrate_too_few_raises(self, calibration_mod):
        calibrator = calibration_mod.HandEyeCalibrator()
        R = np.eye(3)
        t = np.zeros(3)
        T = calibration_mod.make_transform(R, t)
        calibrator.add_pose(calibration_mod.CalibrationPose(T_base_to_ee=T, T_camera_to_target=T))
        with pytest.raises(ValueError, match="Need >= 3"):
            calibrator.calibrate()

    def test_calibrate_returns_valid_result(self, calibration_mod):
        calibrator = calibration_mod.HandEyeCalibrator()
        poses = self._generate_synthetic_poses(calibration_mod, n_poses=20)
        for p in poses:
            calibrator.add_pose(p)
        result = calibrator.calibrate()
        assert result.T_hand_eye.shape == (4, 4)
        assert result.n_poses == 20
        assert result.mean_translation_error_mm >= 0
        assert result.mean_rotation_error_deg >= 0
        assert isinstance(result.residuals, list)


# ── PatientRegistration ───────────────────────────────────────────────────


class TestPatientRegistration:
    def test_registration_identity(self, calibration_mod):
        reg = calibration_mod.PatientRegistration()
        pts = np.random.randn(10, 3) * 0.1
        for i in range(len(pts)):
            reg.add_fiducial(
                name=f"fid_{i}",
                imaging_position_m=pts[i],
                physical_position_m=pts[i],
            )
        result = reg.register_fiducial()
        np.testing.assert_array_almost_equal(result.T_imaging_to_robot, np.eye(4), decimal=5)

    def test_registration_known_transform(self, calibration_mod):
        reg = calibration_mod.PatientRegistration()
        R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        t = np.array([0.1, 0.2, 0.0])
        T_gt = calibration_mod.make_transform(R, t)

        pts_img = np.random.randn(8, 3) * 0.1
        pts_phys = calibration_mod.transform_points(T_gt, pts_img)
        for i in range(len(pts_img)):
            reg.add_fiducial(
                name=f"fid_{i}",
                imaging_position_m=pts_img[i],
                physical_position_m=pts_phys[i],
            )
        result = reg.register_fiducial()
        assert result.fre_mm < 1.0

    def test_registration_too_few_raises(self, calibration_mod):
        reg = calibration_mod.PatientRegistration()
        reg.add_fiducial(
            name="only_one",
            imaging_position_m=np.zeros(3),
            physical_position_m=np.zeros(3),
        )
        with pytest.raises(ValueError, match="Need >= 3"):
            reg.register_fiducial()


# ── CalibrationVerifier ───────────────────────────────────────────────────


class TestCalibrationVerifier:
    def test_verify_with_identity_transforms(self, calibration_mod):
        verifier = calibration_mod.CalibrationVerifier(
            T_hand_eye=np.eye(4),
            T_imaging_to_robot=np.eye(4),
        )
        test_points = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        for i, pt in enumerate(test_points):
            verifier.add_test_point(
                name=f"tp_{i}",
                imaging_position_m=pt,
                physical_position_m=pt,
            )
        result = verifier.verify()
        assert result["mean_error_mm"] < 0.1
        assert result["max_error_mm"] < 0.1
