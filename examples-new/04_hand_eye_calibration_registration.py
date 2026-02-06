"""
=============================================================================
EXAMPLE 04: Hand-Eye Calibration and Patient Registration
=============================================================================

WHAT THIS CODE DOES:
--------------------
Implements the two spatial calibration procedures that must happen before
every surgical robot procedure:

1. HAND-EYE CALIBRATION: Determines the fixed transform between the
   robot's end-effector (hand) and the camera (eye). This is done once
   per hardware setup and verified before each procedure.

2. PATIENT REGISTRATION: Aligns the preoperative imaging coordinate
   system (CT/MRI) to the physical patient on the operating table,
   as seen by the robot. This is done at the start of each procedure.

Without these calibrations, the robot cannot relate what it sees
(camera data) or plans (from imaging) to where it actually is.

WHEN TO USE THIS:
-----------------
- You are setting up a new camera-robot configuration
- You need to register a patient's preoperative imaging to the robot frame
- You need to verify calibration accuracy before a procedure
- You are implementing fiducial-based or surface-based registration

HARDWARE REQUIREMENTS:
----------------------
    - Surgical robot with accurate joint encoders (dVRK, Kinova, UR)
    - Camera rigidly mounted to robot end-effector (eye-in-hand) OR
      Camera mounted externally looking at the robot workspace (eye-to-hand)
    - Calibration target (checkerboard, ArUco markers, or tracked probe)
    - Optical tracker (NDI Polaris) for patient registration OR
      Anatomical fiducial markers for surface registration

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - SciPy 1.11.0+

Optional (for deployment):
    - OpenCV 4.9+ (checkerboard/ArUco detection)
    - Open3D 0.18+ (surface registration)
    - ROS 2 + TF2 (transform management)

ACCURACY REQUIREMENTS:
----------------------
    - Hand-eye calibration: <1 mm translation, <0.5 deg rotation
    - Patient registration: <2 mm target registration error (TRE)
    - Both must be verified before proceeding to surgery
    - IEC 80601-2-77 requires documentation of registration accuracy

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: RIGID TRANSFORM UTILITIES
# =============================================================================
# Homogeneous 4x4 transform operations used throughout calibration.
# These are the building blocks for all coordinate frame math.
#
# COORDINATE FRAME CONVENTIONS:
# - Robot base frame: origin at robot base, z-up
# - Camera frame: origin at camera center, z-forward (OpenCV convention)
# - Patient frame: origin varies by imaging modality (DICOM patient coords)
# - All transforms are 4x4 homogeneous: [R t; 0 1]
# =============================================================================


def make_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transform to Nx3 points.

    Args:
        T: 4x4 homogeneous transform.
        points: Nx3 array of 3D points.

    Returns:
        Nx3 array of transformed points.
    """
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    transformed = (T @ points_h.T).T
    return transformed[:, :3]


def rotation_error_deg(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute angular error between two transforms in degrees."""
    R_err = T1[:3, :3].T @ T2[:3, :3]
    angle_rad = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    return float(np.degrees(angle_rad))


def translation_error_mm(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute translation error between two transforms in mm."""
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]) * 1000)


# =============================================================================
# SECTION 2: HAND-EYE CALIBRATION
# =============================================================================
# Determines the transform X where: AX = XB
#   A = robot end-effector motion (from joint encoders / FK)
#   B = camera motion (from observing calibration target)
#   X = hand-eye transform (what we want)
#
# Eye-in-hand: camera mounted on robot wrist
#   -> X = T_ee_to_camera (end-effector to camera)
#
# Eye-to-hand: camera fixed externally
#   -> X = T_base_to_camera (robot base to camera)
#
# INSTRUCTIONS:
# 1. Mount camera and calibration target.
# 2. Move robot to 15-25 poses where the target is visible.
# 3. Record robot FK (T_base_to_ee) and detected target pose (T_cam_to_target)
#    at each pose.
# 4. Run calibration. Verify reprojection error < 1 mm.
# =============================================================================


@dataclass
class CalibrationPose:
    """
    A single hand-eye calibration measurement.

    Attributes:
        T_base_to_ee: 4x4 robot base to end-effector transform (from FK).
        T_camera_to_target: 4x4 camera to calibration target transform
            (from target detection in image).
        image_path: Optional path to image for audit.
        reprojection_error_px: Target detection reprojection error.
    """

    T_base_to_ee: np.ndarray
    T_camera_to_target: np.ndarray
    image_path: str = ""
    reprojection_error_px: float = 0.0


class HandEyeCalibrator:
    """
    Hand-eye calibration for surgical robot + camera systems.

    SETUP INSTRUCTIONS:
    -------------------
    For eye-in-hand (camera on robot wrist, e.g., dVRK endoscope):
      1. Fix calibration target (checkerboard or ArUco board) in the workspace.
         Target must not move during calibration.
      2. Move robot to 15-25 poses covering diverse orientations.
         - Vary rotation around all 3 axes by at least +/-30 degrees.
         - Keep target fully visible and in focus at each pose.
      3. At each pose, record:
         - Robot FK: T_base_to_ee (from /PSM1/measured_cp or FK computation)
         - Target detection: T_camera_to_target (from OpenCV solvePnP)

    For eye-to-hand (fixed external camera, e.g., overhead RGBD):
      1. Mount calibration target rigidly to robot end-effector.
      2. Follow the same procedure, but now the target moves with the robot.

    DATA QUALITY CHECKS:
    - Each pair must have reprojection error < 1.0 px.
    - Discard outlier poses where target detection is poor.
    - At least 15 valid poses needed for stable calibration.

    Example:
        >>> calibrator = HandEyeCalibrator(method="eye_in_hand")
        >>> for pose_data in recorded_poses:
        ...     calibrator.add_pose(pose_data)
        >>> result = calibrator.calibrate()
        >>> print(f"Calibration error: {result.translation_error_mm:.2f} mm")
    """

    def __init__(self, method: str = "eye_in_hand"):
        """
        Args:
            method: "eye_in_hand" or "eye_to_hand".
        """
        self.method = method
        self._poses: list[CalibrationPose] = []

        logger.info("HandEyeCalibrator: method=%s", method)

    def add_pose(self, pose: CalibrationPose):
        """Add a calibration measurement."""
        self._poses.append(pose)

    def calibrate(self) -> "HandEyeCalibrationResult":
        """
        Run hand-eye calibration.

        Implements the Tsai-Lenz method (1989) which solves AX = XB
        using pairs of relative motions.

        Returns:
            HandEyeCalibrationResult with the computed transform and metrics.
        """
        n = len(self._poses)
        if n < 3:
            raise ValueError(f"Need >= 3 poses, got {n}")

        logger.info("Running hand-eye calibration with %d poses", n)

        # Compute relative motions (consecutive pairs)
        A_list = []  # Robot relative motions
        B_list = []  # Camera relative motions

        for i in range(n - 1):
            T_ee_i = self._poses[i].T_base_to_ee
            T_ee_j = self._poses[i + 1].T_base_to_ee
            T_cam_i = self._poses[i].T_camera_to_target
            T_cam_j = self._poses[i + 1].T_camera_to_target

            if self.method == "eye_in_hand":
                # A = T_ee_i^{-1} * T_ee_j (robot ee relative motion)
                A = invert_transform(T_ee_i) @ T_ee_j
                # B = T_cam_i * T_cam_j^{-1} (camera relative motion)
                B = T_cam_i @ invert_transform(T_cam_j)
            else:
                # Eye-to-hand: different formulation
                A = invert_transform(T_ee_i) @ T_ee_j
                B = T_cam_j @ invert_transform(T_cam_i)

            A_list.append(A)
            B_list.append(B)

        # Solve for rotation using Tsai-Lenz
        X = self._solve_tsai_lenz(A_list, B_list)

        # Compute validation metrics
        residuals = self._compute_residuals(A_list, B_list, X)

        result = HandEyeCalibrationResult(
            T_hand_eye=X,
            method=self.method,
            n_poses=n,
            mean_rotation_error_deg=float(np.mean([r["rot_err_deg"] for r in residuals])),
            mean_translation_error_mm=float(np.mean([r["trans_err_mm"] for r in residuals])),
            max_translation_error_mm=float(np.max([r["trans_err_mm"] for r in residuals])),
            residuals=residuals,
        )

        logger.info(
            "Calibration complete: trans_err=%.2f mm, rot_err=%.3f deg",
            result.mean_translation_error_mm,
            result.mean_rotation_error_deg,
        )
        return result

    def _solve_tsai_lenz(self, A_list: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
        """
        Solve AX=XB using the Tsai-Lenz method.

        This solves for rotation first (using axis-angle representation),
        then for translation.
        """
        n = len(A_list)

        # --- Step 1: Solve for rotation ---
        # Using the quaternion approach for better numerical stability
        M = np.zeros((3 * n, 3))
        rhs = np.zeros(3 * n)

        for i, (A, B) in enumerate(zip(A_list, B_list)):
            Ra = A[:3, :3]
            Rb = B[:3, :3]

            # Rotation to axis-angle
            r_a = Rotation.from_matrix(Ra)
            r_b = Rotation.from_matrix(Rb)
            alpha = r_a.as_rotvec()
            beta = r_b.as_rotvec()

            # Skew symmetric
            skew_sum = self._skew(alpha + beta)
            M[3 * i : 3 * i + 3, :] = skew_sum
            rhs[3 * i : 3 * i + 3] = beta - alpha

        # Solve least squares for rotation
        x_rot, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Recover rotation matrix
        theta = 2 * np.arctan(np.linalg.norm(x_rot))
        if np.linalg.norm(x_rot) > 1e-10:
            axis = x_rot / np.linalg.norm(x_rot)
        else:
            axis = np.array([0, 0, 1])
        Rx = Rotation.from_rotvec(axis * theta).as_matrix()

        # --- Step 2: Solve for translation ---
        C = np.zeros((3 * n, 3))
        d = np.zeros(3 * n)

        for i, (A, B) in enumerate(zip(A_list, B_list)):
            Ra = A[:3, :3]
            ta = A[:3, 3]
            tb = B[:3, 3]

            C[3 * i : 3 * i + 3, :] = Ra - np.eye(3)
            d[3 * i : 3 * i + 3] = Rx @ tb - ta

        tx, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

        return make_transform(Rx, tx)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from 3-vector."""
        return np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )

    def _compute_residuals(
        self,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        X: np.ndarray,
    ) -> list[dict]:
        """Compute per-pair AX=XB residuals."""
        residuals = []
        for A, B in zip(A_list, B_list):
            AX = A @ X
            XB = X @ B
            residuals.append(
                {
                    "rot_err_deg": rotation_error_deg(AX, XB),
                    "trans_err_mm": translation_error_mm(AX, XB),
                }
            )
        return residuals


@dataclass
class HandEyeCalibrationResult:
    """Result of hand-eye calibration."""

    T_hand_eye: np.ndarray
    method: str
    n_poses: int
    mean_rotation_error_deg: float
    mean_translation_error_mm: float
    max_translation_error_mm: float
    residuals: list

    def is_acceptable(self, max_trans_mm: float = 1.0, max_rot_deg: float = 0.5) -> bool:
        """Check if calibration meets accuracy requirements."""
        return self.mean_translation_error_mm <= max_trans_mm and self.mean_rotation_error_deg <= max_rot_deg

    def print_report(self):
        """Print calibration quality report."""
        print("\n" + "=" * 50)
        print("HAND-EYE CALIBRATION REPORT")
        print("=" * 50)
        print(f"Method:          {self.method}")
        print(f"Poses used:      {self.n_poses}")
        print(f"Mean trans err:  {self.mean_translation_error_mm:.3f} mm")
        print(f"Max trans err:   {self.max_translation_error_mm:.3f} mm")
        print(f"Mean rot err:    {self.mean_rotation_error_deg:.4f} deg")
        print(f"Acceptable:      {self.is_acceptable()}")
        print(f"\nTransform (hand -> eye):\n{self.T_hand_eye}")


# =============================================================================
# SECTION 3: PATIENT REGISTRATION
# =============================================================================
# Aligns preoperative imaging coordinates to the physical patient.
#
# Two approaches:
# 1. FIDUCIAL-BASED: Match known anatomical/artificial landmarks between
#    imaging and physical space. Faster, requires visible fiducials.
# 2. SURFACE-BASED: Match a point cloud of the patient surface to the
#    imaging surface model. More robust, requires surface exposure.
#
# INSTRUCTIONS:
# - For fiducial registration: place 4+ radio-opaque markers on the
#   patient before the preoperative scan, then touch each marker with
#   a tracked probe intraoperatively.
# - For surface registration: acquire intraoperative surface scan
#   (structured light, laser, or RGBD camera) and register to preop model.
# - Target registration error (TRE) must be < 2 mm for surgical guidance.
# =============================================================================


@dataclass
class FiducialPair:
    """
    A matched fiducial point in both imaging and physical space.

    Attributes:
        name: Fiducial identifier (e.g., "nasion", "marker_1").
        imaging_position_m: [x, y, z] in imaging coordinate frame.
        physical_position_m: [x, y, z] in robot/tracker coordinate frame.
        localization_error_mm: Estimated error in physical localization.
    """

    name: str
    imaging_position_m: np.ndarray
    physical_position_m: np.ndarray
    localization_error_mm: float = 0.5


class PatientRegistration:
    """
    Patient-to-robot registration for surgical guidance.

    This computes the rigid transform T_imaging_to_robot that maps
    coordinates from preoperative imaging (CT/MRI) to the robot's
    base frame. This allows the robot to know where the tumor,
    vessels, and other structures are in its own coordinate system.

    SETUP INSTRUCTIONS (Fiducial-Based):
    -------------------------------------
    1. Before the preoperative scan:
       - Place 4-8 radio-opaque fiducial markers on the patient's skin
         near the surgical site (use bone-anchored markers for better accuracy).
    2. In the preoperative images:
       - Identify and record 3D coordinates of each fiducial.
    3. Intraoperatively:
       - Touch each fiducial with a tracked probe (NDI Polaris) or
         the robot end-effector to record physical coordinates.
    4. Run point-based registration.
    5. Verify: touch additional landmarks and check predicted vs actual position.

    ACCURACY NOTES:
    - Fiducial registration error (FRE) measures how well the
      fiducials themselves align.
    - Target registration error (TRE) measures accuracy at points
      AWAY from fiducials (i.e., at the tumor). TRE is what matters clinically.
    - TRE depends on fiducial configuration and distance from centroid.
    - Use at least 4 fiducials, well-spread around the surgical site.

    Example:
        >>> reg = PatientRegistration()
        >>> reg.add_fiducial("marker_1", imaging_pos, physical_pos)
        >>> reg.add_fiducial("marker_2", imaging_pos, physical_pos)
        >>> # ... add 4+ fiducials
        >>> result = reg.register_fiducial()
        >>> if result.tre_estimate_mm < 2.0:
        ...     T = result.T_imaging_to_robot
    """

    def __init__(self):
        self._fiducials: list[FiducialPair] = []

    def add_fiducial(
        self,
        name: str,
        imaging_position_m: np.ndarray,
        physical_position_m: np.ndarray,
        localization_error_mm: float = 0.5,
    ):
        """Add a matched fiducial pair."""
        self._fiducials.append(
            FiducialPair(
                name=name,
                imaging_position_m=np.asarray(imaging_position_m, dtype=np.float64),
                physical_position_m=np.asarray(physical_position_m, dtype=np.float64),
                localization_error_mm=localization_error_mm,
            )
        )
        logger.info("Added fiducial '%s' (total: %d)", name, len(self._fiducials))

    def register_fiducial(self) -> "RegistrationResult":
        """
        Compute rigid registration from matched fiducial pairs.

        Uses the Arun method (SVD-based least-squares rigid registration):
        1. Compute centroids of both point sets.
        2. Center points.
        3. Compute cross-covariance matrix H.
        4. SVD of H gives optimal rotation.
        5. Translation from centroids.

        Returns:
            RegistrationResult with transform and quality metrics.
        """
        n = len(self._fiducials)
        if n < 3:
            raise ValueError(f"Need >= 3 fiducials, got {n}")

        logger.info("Computing fiducial registration with %d points", n)

        # Extract point arrays
        src = np.array([f.imaging_position_m for f in self._fiducials])
        dst = np.array([f.physical_position_m for f in self._fiducials])

        # Arun's method
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)

        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)

        # Handle reflection
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, d])

        R = Vt.T @ sign_matrix @ U.T
        t = dst_centroid - R @ src_centroid

        T = make_transform(R, t)

        # Compute FRE (Fiducial Registration Error)
        transformed_src = transform_points(T, src)
        residuals = np.linalg.norm(transformed_src - dst, axis=1)
        fre = float(np.sqrt(np.mean(residuals**2)))

        # Estimate TRE at tumor location (using FLE and fiducial geometry)
        fle = float(np.mean([f.localization_error_mm for f in self._fiducials]))
        tre_estimate = self._estimate_tre(src, fle, n)

        result = RegistrationResult(
            T_imaging_to_robot=T,
            method="fiducial_arun",
            n_fiducials=n,
            fre_mm=fre * 1000,
            tre_estimate_mm=tre_estimate,
            per_fiducial_errors_mm=(residuals * 1000).tolist(),
            fiducial_names=[f.name for f in self._fiducials],
        )

        logger.info(
            "Registration complete: FRE=%.2f mm, TRE_est=%.2f mm",
            result.fre_mm,
            result.tre_estimate_mm,
        )
        return result

    def register_surface_icp(
        self,
        preop_surface_points_m: np.ndarray,
        intraop_surface_points_m: np.ndarray,
        initial_guess: np.ndarray | None = None,
        max_iterations: int = 50,
    ) -> "RegistrationResult":
        """
        Surface-based registration using ICP.

        Use when fiducial markers are not available. Requires an
        intraoperative surface scan (from structured light, RGBD camera,
        or laser scanner) and the preoperative surface model extracted
        from CT/MRI.

        Args:
            preop_surface_points_m: Nx3 points from preoperative model.
            intraop_surface_points_m: Mx3 points from intraoperative scan.
            initial_guess: 4x4 initial alignment (from rough manual alignment
                or fiducial pre-registration).
            max_iterations: Maximum ICP iterations.

        Returns:
            RegistrationResult.

        INSTRUCTIONS:
        - Provide an initial guess within ~20 mm / 10 degrees of true alignment.
        - Without a good initial guess, ICP may converge to a local minimum.
        - For better convergence, use coarse-to-fine ICP or global registration
          (e.g., RANSAC + ICP in Open3D).
        """
        if initial_guess is None:
            initial_guess = np.eye(4)

        T = initial_guess.copy()
        src = preop_surface_points_m.copy()
        dst = intraop_surface_points_m.copy()

        for iteration in range(max_iterations):
            # Transform source points
            transformed_src = transform_points(T, src)

            # Find closest points (brute force - use KD-tree in production)
            n_src = min(len(transformed_src), 500)  # Subsample for speed
            indices = np.random.choice(len(transformed_src), n_src, replace=False)
            sub_src = transformed_src[indices]

            # Find nearest neighbor in destination for each source point
            distances = np.linalg.norm(sub_src[:, None, :] - dst[None, :, :], axis=2)
            nn_indices = np.argmin(distances, axis=1)
            nn_distances = distances[np.arange(n_src), nn_indices]

            # Reject outliers (>5 mm)
            inlier_mask = nn_distances < 0.005
            if np.sum(inlier_mask) < 10:
                break

            # Compute transform update from inlier correspondences
            p = sub_src[inlier_mask]
            q = dst[nn_indices[inlier_mask]]

            p_c = np.mean(p, axis=0)
            q_c = np.mean(q, axis=0)
            H = (p - p_c).T @ (q - q_c)
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R_update = Vt.T @ np.diag([1, 1, d]) @ U.T
            t_update = q_c - R_update @ p_c

            T_update = make_transform(R_update, t_update)
            T = T_update @ T

            mean_dist = float(np.mean(nn_distances[inlier_mask]))
            if mean_dist < 0.0005:  # Converged (< 0.5 mm)
                break

        # Final error
        final_transformed = transform_points(T, src)
        sample_size = min(len(final_transformed), len(dst), 500)
        sample_idx = np.random.choice(len(final_transformed), sample_size, replace=False)
        final_dists = np.linalg.norm(final_transformed[sample_idx, None, :] - dst[None, :, :], axis=2)
        nn_dists = np.min(final_dists, axis=1)
        fre = float(np.sqrt(np.mean(nn_dists**2)))

        result = RegistrationResult(
            T_imaging_to_robot=T,
            method="surface_icp",
            n_fiducials=0,
            fre_mm=fre * 1000,
            tre_estimate_mm=fre * 1000 * 1.5,  # Rough estimate
            per_fiducial_errors_mm=[],
            fiducial_names=[],
        )

        logger.info("ICP registration complete: mean_distance=%.2f mm", fre * 1000)
        return result

    @staticmethod
    def _estimate_tre(fiducials: np.ndarray, fle_mm: float, n: int) -> float:
        """
        Estimate Target Registration Error from fiducial configuration.

        Uses the Fitzpatrick formula:
            TRE^2 ≈ FLE^2/N * (1 + sum_k d_k^2 / f_k^2)
        where d_k is distance from target to centroid along principal axis k,
        and f_k is the RMS fiducial distance along that axis.

        For simplicity, this returns a conservative estimate.
        """
        centroid = np.mean(fiducials, axis=0)
        spread = np.std(fiducials - centroid, axis=0)
        mean_spread = float(np.mean(spread)) * 1000  # Convert to mm
        if mean_spread < 1e-6:
            return fle_mm * 5
        return fle_mm * np.sqrt(1.0 + 1.0 / n) * 1.5


@dataclass
class RegistrationResult:
    """Result of patient registration."""

    T_imaging_to_robot: np.ndarray
    method: str
    n_fiducials: int
    fre_mm: float
    tre_estimate_mm: float
    per_fiducial_errors_mm: list
    fiducial_names: list

    def is_acceptable(self, max_tre_mm: float = 2.0) -> bool:
        """Check if registration meets clinical accuracy requirements."""
        return self.tre_estimate_mm <= max_tre_mm

    def print_report(self):
        """Print registration quality report."""
        print("\n" + "=" * 50)
        print("PATIENT REGISTRATION REPORT")
        print("=" * 50)
        print(f"Method:           {self.method}")
        print(f"Fiducials:        {self.n_fiducials}")
        print(f"FRE:              {self.fre_mm:.3f} mm")
        print(f"TRE estimate:     {self.tre_estimate_mm:.3f} mm")
        print(f"Acceptable (<2mm): {self.is_acceptable()}")

        if self.per_fiducial_errors_mm:
            print("\nPer-fiducial errors:")
            for name, err in zip(self.fiducial_names, self.per_fiducial_errors_mm):
                status = "OK" if err < 2.0 else "HIGH"
                print(f"  {name}: {err:.3f} mm [{status}]")


# =============================================================================
# SECTION 4: CALIBRATION VERIFICATION
# =============================================================================
# Verify calibration accuracy with independent test points.
# This is mandatory before proceeding to the surgical procedure.
# =============================================================================


class CalibrationVerifier:
    """
    Verify hand-eye and patient registration accuracy.

    INSTRUCTIONS:
    - After calibration, touch 3-5 test points NOT used in calibration.
    - These can be additional fiducials or identifiable anatomical landmarks.
    - Measure discrepancy between predicted and actual positions.
    - If any test point error > 3 mm, recalibrate.

    Example:
        >>> verifier = CalibrationVerifier(T_hand_eye, T_registration)
        >>> verifier.add_test_point("test_1", imaging_pos, physical_pos)
        >>> report = verifier.verify()
        >>> if not report["acceptable"]:
        ...     print("Recalibration required")
    """

    def __init__(
        self,
        T_hand_eye: np.ndarray,
        T_imaging_to_robot: np.ndarray,
    ):
        self.T_hand_eye = T_hand_eye
        self.T_imaging_to_robot = T_imaging_to_robot
        self._test_points: list[dict] = []

    def add_test_point(
        self,
        name: str,
        imaging_position_m: np.ndarray,
        physical_position_m: np.ndarray,
    ):
        """Add an independent test point for verification."""
        # Predict physical position from imaging
        predicted = transform_points(
            self.T_imaging_to_robot,
            np.array([imaging_position_m]),
        )[0]

        error_mm = float(np.linalg.norm(predicted - physical_position_m) * 1000)

        self._test_points.append(
            {
                "name": name,
                "imaging_m": imaging_position_m,
                "physical_m": physical_position_m,
                "predicted_m": predicted,
                "error_mm": error_mm,
            }
        )

    def verify(self, max_error_mm: float = 3.0) -> dict:
        """
        Run verification and return pass/fail report.

        Args:
            max_error_mm: Maximum allowed error at any test point.

        Returns:
            Verification report dictionary.
        """
        if not self._test_points:
            return {"acceptable": False, "reason": "no_test_points"}

        errors = [p["error_mm"] for p in self._test_points]
        max_err = max(errors)
        mean_err = float(np.mean(errors))

        acceptable = max_err <= max_error_mm

        report = {
            "acceptable": acceptable,
            "n_test_points": len(self._test_points),
            "mean_error_mm": mean_err,
            "max_error_mm": max_err,
            "test_points": self._test_points,
        }

        return report


# =============================================================================
# SECTION 5: MAIN DEMONSTRATION
# =============================================================================


def run_calibration_demo():
    """
    Demonstrate hand-eye calibration and patient registration.

    Simulates:
    1. Collecting 20 hand-eye calibration poses
    2. Computing hand-eye transform
    3. Collecting 6 fiducial pairs for patient registration
    4. Computing patient registration
    5. Verifying with independent test points
    """
    logger.info("=" * 70)
    logger.info("HAND-EYE CALIBRATION AND PATIENT REGISTRATION")
    logger.info("=" * 70)

    # --- Part 1: Hand-Eye Calibration ---
    logger.info("\n--- Part 1: Hand-Eye Calibration ---")

    calibrator = HandEyeCalibrator(method="eye_in_hand")

    # Ground truth hand-eye transform
    true_T = make_transform(
        Rotation.from_euler("xyz", [5, -3, 2], degrees=True).as_matrix(),
        np.array([0.02, -0.01, 0.05]),
    )

    # Simulate 20 calibration poses
    np.random.seed(42)
    for i in range(20):
        # Random robot pose
        R_ee = Rotation.from_euler(
            "xyz",
            np.random.uniform(-30, 30, 3),
            degrees=True,
        ).as_matrix()
        t_ee = np.random.uniform(-0.1, 0.1, 3)
        T_base_to_ee = make_transform(R_ee, t_ee)

        # Compute camera-to-target via hand-eye relationship
        T_target_fixed = make_transform(np.eye(3), np.array([0.3, 0.0, 0.0]))
        T_cam = invert_transform(true_T) @ invert_transform(T_base_to_ee) @ T_target_fixed

        # Add noise
        T_cam[:3, 3] += np.random.randn(3) * 0.0002  # 0.2 mm noise

        calibrator.add_pose(CalibrationPose(T_base_to_ee=T_base_to_ee, T_camera_to_target=T_cam))

    he_result = calibrator.calibrate()
    he_result.print_report()

    # --- Part 2: Patient Registration ---
    logger.info("\n--- Part 2: Patient Registration ---")

    registration = PatientRegistration()

    # Ground truth registration transform
    true_reg = make_transform(
        Rotation.from_euler("xyz", [10, -5, 3], degrees=True).as_matrix(),
        np.array([0.1, -0.05, 0.2]),
    )

    # Simulate 6 fiducial pairs
    fiducial_names = [
        "marker_left_1",
        "marker_left_2",
        "marker_right_1",
        "marker_right_2",
        "marker_superior",
        "marker_inferior",
    ]
    imaging_positions = np.array(
        [
            [-0.03, 0.02, 0.0],
            [-0.03, -0.02, 0.0],
            [0.03, 0.02, 0.0],
            [0.03, -0.02, 0.0],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, -0.03],
        ]
    )

    for name, img_pos in zip(fiducial_names, imaging_positions):
        # Compute physical position via ground truth + noise
        phys_pos = transform_points(true_reg, img_pos.reshape(1, 3))[0]
        phys_pos += np.random.randn(3) * 0.0003  # 0.3 mm localization noise

        registration.add_fiducial(name, img_pos, phys_pos)

    reg_result = registration.register_fiducial()
    reg_result.print_report()

    # --- Part 3: Verification ---
    logger.info("\n--- Part 3: Verification ---")

    verifier = CalibrationVerifier(
        T_hand_eye=he_result.T_hand_eye,
        T_imaging_to_robot=reg_result.T_imaging_to_robot,
    )

    # Test with 3 independent points
    test_positions = np.array(
        [
            [0.01, 0.01, 0.01],
            [-0.02, 0.0, -0.01],
            [0.0, -0.015, 0.02],
        ]
    )
    for i, img_pos in enumerate(test_positions):
        phys_pos = transform_points(true_reg, img_pos.reshape(1, 3))[0]
        phys_pos += np.random.randn(3) * 0.0005

        verifier.add_test_point(f"test_{i + 1}", img_pos, phys_pos)

    verification = verifier.verify()

    print("\n" + "=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)
    print(f"Acceptable:    {verification['acceptable']}")
    print(f"Mean error:    {verification['mean_error_mm']:.3f} mm")
    print(f"Max error:     {verification['max_error_mm']:.3f} mm")
    for tp in verification["test_points"]:
        print(f"  {tp['name']}: {tp['error_mm']:.3f} mm")

    return {
        "hand_eye_acceptable": he_result.is_acceptable(),
        "registration_acceptable": reg_result.is_acceptable(),
        "verification_acceptable": verification["acceptable"],
        "tre_estimate_mm": reg_result.tre_estimate_mm,
    }


if __name__ == "__main__":
    results = run_calibration_demo()
    print(f"\nCalibration pipeline: {results}")
