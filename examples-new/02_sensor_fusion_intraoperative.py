"""
=============================================================================
EXAMPLE 02: Multi-Sensor Fusion for Intraoperative Perception
=============================================================================

WHAT THIS CODE DOES:
--------------------
Implements the real-time perception pipeline that runs on a physical
surgical robot during oncology procedures. This system fuses data from
multiple sensors (stereo cameras, depth sensors, force-torque, joint
encoders, and tracked instruments) into a unified scene representation
that the robot controller and surgeon can act on.

WHEN TO USE THIS:
-----------------
- You are building the perception stack for a surgical robot
- You need real-time instrument tracking during tumor resection
- You must fuse stereo endoscope imagery with force sensing
- You are implementing tissue deformation tracking for margin assessment
- You need point cloud registration between preop imaging and live anatomy

HARDWARE REQUIREMENTS:
----------------------
    - Stereo endoscope or RGBD camera (e.g., Intel RealSense D405 for
      external view, or da Vinci stereo endoscope for intracorporeal)
    - Force-torque sensor at instrument tip (see example 01)
    - Optical tracker (NDI Polaris or similar) for patient registration
    - GPU for real-time inference (NVIDIA RTX 3060+ recommended)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - SciPy 1.11.0+
    - Python 3.10+

Optional (for deployment):
    - OpenCV 4.9+ (stereo processing)
    - Open3D 0.18+ (point cloud processing)
    - PyTorch 2.5+ (neural network inference)
    - MONAI 1.4+ (medical image segmentation)
    - ROS 2 Jazzy (sensor drivers, TF2 transforms)

CLINICAL CONTEXT:
-----------------
During robotic tumor resection, the surgeon needs:
    - Clear visualization of tumor margins (where healthy tissue begins)
    - Knowledge of instrument tip position relative to critical structures
    - Deformation tracking as tissue shifts during manipulation
    - Force feedback to detect unexpected contact

This perception pipeline provides all of these by combining sensor data
at rates sufficient for real-time surgical guidance (>15 Hz for visual,
>100 Hz for force).

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: SENSOR DATA TYPES
# =============================================================================
# Each sensor produces a specific data type at its native rate.
# The fusion pipeline synchronizes these into a common time frame.
#
# INSTRUCTIONS:
# - Each dataclass corresponds to a ROS 2 message type in production.
# - Timestamps must use the same monotonic clock across all sensors.
# - For dVRK, use the cisst/SAW clock. For ROS 2, use rclcpp::Clock.
# =============================================================================


class SensorType(Enum):
    STEREO_CAMERA = auto()
    DEPTH_CAMERA = auto()
    FORCE_TORQUE = auto()
    JOINT_ENCODER = auto()
    OPTICAL_TRACKER = auto()
    ENDOSCOPE = auto()


@dataclass
class StereoImagePair:
    """
    Synchronized stereo image pair from endoscope or external cameras.

    SETUP INSTRUCTIONS:
    - Calibrate stereo pair using OpenCV's stereoCalibrate with a
      checkerboard (at least 20 image pairs at varied orientations).
    - Store calibration as camera_matrix_left, camera_matrix_right,
      distortion_coeffs, R (rotation), T (translation), Q (disparity-to-depth).
    - Re-calibrate after autoclaving if using reusable endoscope.

    Attributes:
        left: Left image as HxWx3 uint8 array (BGR).
        right: Right image as HxWx3 uint8 array (BGR).
        timestamp_ns: Capture timestamp (both images synchronized).
        camera_matrix_left: 3x3 intrinsic matrix for left camera.
        camera_matrix_right: 3x3 intrinsic matrix for right camera.
        baseline_m: Stereo baseline in meters.
    """

    left: np.ndarray
    right: np.ndarray
    timestamp_ns: int
    camera_matrix_left: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )
    camera_matrix_right: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )
    baseline_m: float = 0.005


@dataclass
class DepthImage:
    """
    Depth image from RGBD camera or computed from stereo.

    Attributes:
        depth_m: HxW float32 array, depth in meters. 0 = invalid.
        rgb: HxWx3 uint8 array (aligned color image).
        timestamp_ns: Capture timestamp.
        camera_matrix: 3x3 intrinsic matrix.
    """

    depth_m: np.ndarray
    rgb: np.ndarray
    timestamp_ns: int
    camera_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class PointCloud:
    """
    3D point cloud in a specified coordinate frame.

    Attributes:
        points_m: Nx3 float32 array of 3D points.
        colors: Nx3 uint8 array of RGB colors (or None).
        normals: Nx3 float32 array of surface normals (or None).
        frame_id: Coordinate frame (e.g., "camera", "robot_base", "patient").
        timestamp_ns: Timestamp.
    """

    points_m: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    frame_id: str = "camera"
    timestamp_ns: int = 0


@dataclass
class TrackedPose:
    """
    Pose from optical tracker (NDI Polaris, Atracsys, etc.).

    Attributes:
        position_m: [x, y, z] in tracker frame.
        orientation_quat: [w, x, y, z] quaternion.
        tool_id: Tracked tool identifier.
        is_visible: Whether the tool is in the tracker's field of view.
        rms_error_mm: Tracking RMS error.
        timestamp_ns: Measurement timestamp.
    """

    position_m: np.ndarray
    orientation_quat: np.ndarray
    tool_id: str
    is_visible: bool = True
    rms_error_mm: float = 0.25
    timestamp_ns: int = 0


# =============================================================================
# SECTION 2: INSTRUMENT SEGMENTATION
# =============================================================================
# Segment surgical instruments from endoscope imagery in real-time.
# This is critical for tracking instrument tip position relative to tissue.
#
# INSTRUCTIONS:
# - Use a pretrained model (MONAI, SAM, or custom U-Net) fine-tuned on
#   surgical instrument datasets (EndoVis, CholecSeg8k, AutoLaparo).
# - Target inference time: <30 ms on RTX 3060 for 640x480 input.
# - Output: per-pixel mask of instrument vs background vs tissue.
# =============================================================================


class InstrumentSegmenter:
    """
    Real-time surgical instrument segmentation from endoscope images.

    SETUP INSTRUCTIONS:
    -------------------
    1. Download pretrained model weights (SegFormer or U-Net trained on
       EndoVis Challenge data, fine-tuned on your instrument set).
    2. Place weights at model_path.
    3. Verify inference speed: must be <30 ms per frame at target resolution.
    4. Validate segmentation IoU >0.85 on held-out test set before clinical use.

    The segmenter outputs class masks:
        0 = background (tissue, anatomy)
        1 = instrument shaft
        2 = instrument tip / end-effector
        3 = instrument wrist / articulation

    Example:
        >>> segmenter = InstrumentSegmenter("weights/instrument_seg.pt")
        >>> mask = segmenter.segment(image_bgr)
        >>> tip_pixels = np.argwhere(mask == 2)
    """

    CLASS_BACKGROUND = 0
    CLASS_SHAFT = 1
    CLASS_TIP = 2
    CLASS_WRIST = 3

    def __init__(self, model_path: str = "", input_size: tuple = (480, 640)):
        self.model_path = model_path
        self.input_size = input_size
        self._model = None
        self._load_model()

    def _load_model(self):
        """
        Load segmentation model.

        In production, replace with:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval().cuda()
        """
        logger.info(
            "InstrumentSegmenter initialized (input_size=%s)", self.input_size
        )

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segment instruments from BGR image.

        Args:
            image_bgr: HxWx3 uint8 BGR image.

        Returns:
            HxW uint8 mask with class labels.
        """
        h, w = image_bgr.shape[:2]
        # Simulated segmentation: in production, run neural network
        mask = np.zeros((h, w), dtype=np.uint8)

        # Simulate instrument in center of image
        cx, cy = w // 2, h // 2
        # Shaft region
        mask[cy - 5 : cy + 5, cx - 100 : cx + 50] = self.CLASS_SHAFT
        # Tip region
        mask[cy - 8 : cy + 8, cx + 50 : cx + 80] = self.CLASS_TIP

        return mask

    def get_tip_position_pixels(self, mask: np.ndarray) -> np.ndarray | None:
        """
        Extract instrument tip centroid in pixel coordinates.

        Returns:
            [u, v] pixel coordinates of tip centroid, or None if not visible.
        """
        tip_pixels = np.argwhere(mask == self.CLASS_TIP)
        if len(tip_pixels) == 0:
            return None
        centroid = np.mean(tip_pixels, axis=0)
        return np.array([centroid[1], centroid[0]])  # (u, v)


# =============================================================================
# SECTION 3: TISSUE TRACKING AND DEFORMATION
# =============================================================================
# Track tissue surface deformation during manipulation.
# Crucial for maintaining awareness of tumor margins as tissue shifts.
#
# INSTRUCTIONS:
# - Initialize tracker from first stable frame (before manipulation starts).
# - Track sparse feature points on tissue surface.
# - When deformation exceeds threshold, re-register with preop model.
# =============================================================================


class TissueDeformationTracker:
    """
    Track tissue surface deformation using point cloud registration.

    This tracks how tissue moves and deforms during surgical manipulation,
    which is critical for maintaining accurate tumor margin awareness.

    SETUP INSTRUCTIONS:
    -------------------
    1. Capture reference point cloud at the start of the procedure
       (before any manipulation).
    2. The reference is registered to preoperative imaging via
       patient registration (see example 04).
    3. During manipulation, each new point cloud is registered to the
       reference to measure deformation.

    Deformation output is used to:
    - Update displayed tumor margin overlay
    - Adjust robot workspace boundaries
    - Warn if deformation exceeds safe limits for margin accuracy
    """

    def __init__(self, max_correspondence_distance_m: float = 0.005):
        """
        Args:
            max_correspondence_distance_m: ICP convergence threshold.
                Set based on expected tissue motion per frame.
                At 15 Hz and max tissue speed 10 mm/s: 0.7 mm per frame,
                so 5 mm threshold provides good margin.
        """
        self.max_correspondence_distance = max_correspondence_distance_m
        self._reference_cloud: PointCloud | None = None
        self._cumulative_deformation_m: float = 0.0
        self._frame_count: int = 0

    def set_reference(self, cloud: PointCloud):
        """
        Set reference point cloud (captured before manipulation).

        INSTRUCTIONS:
        - Capture from stable viewpoint with full surgical field visible.
        - Verify point cloud quality: >1000 points, no large gaps.
        - Store reference for post-procedure analysis.
        """
        self._reference_cloud = cloud
        self._frame_count = 0
        self._cumulative_deformation_m = 0.0
        logger.info(
            "Reference cloud set: %d points in '%s' frame",
            cloud.points_m.shape[0],
            cloud.frame_id,
        )

    def track(self, current_cloud: PointCloud) -> dict:
        """
        Compute deformation between current and reference clouds.

        Uses ICP (Iterative Closest Point) to align clouds and measure
        residual deformation. In production, use Open3D or PCL ICP.

        Args:
            current_cloud: Current point cloud from depth sensor.

        Returns:
            Dictionary with deformation metrics:
            - transform_4x4: 4x4 rigid transform (reference -> current)
            - mean_deformation_m: Mean point displacement after alignment
            - max_deformation_m: Maximum point displacement
            - deformation_field: Per-point displacement vectors
            - is_valid: Whether registration succeeded
        """
        if self._reference_cloud is None:
            return {"is_valid": False, "reason": "no_reference"}

        self._frame_count += 1

        # --- ICP Registration (simplified) ---
        # In production, use Open3D:
        #   result = o3d.pipelines.registration.registration_icp(
        #       current_pcd, reference_pcd,
        #       self.max_correspondence_distance,
        #       np.eye(4),
        #       o3d.pipelines.registration.TransformationEstimationPointToPlane()
        #   )

        ref_points = self._reference_cloud.points_m
        cur_points = current_cloud.points_m

        n_ref = ref_points.shape[0]
        n_cur = cur_points.shape[0]
        n_common = min(n_ref, n_cur)

        # Simulated ICP: compute centroid shift as rigid transform
        ref_centroid = np.mean(ref_points[:n_common], axis=0)
        cur_centroid = np.mean(cur_points[:n_common], axis=0)
        translation = cur_centroid - ref_centroid

        transform = np.eye(4)
        transform[:3, 3] = translation

        # Compute per-point deformation (after rigid alignment)
        aligned = cur_points[:n_common] - translation
        displacements = np.linalg.norm(aligned - ref_points[:n_common], axis=1)
        mean_deformation = float(np.mean(displacements))
        max_deformation = float(np.max(displacements))

        self._cumulative_deformation_m = max(
            self._cumulative_deformation_m, max_deformation
        )

        return {
            "is_valid": True,
            "transform_4x4": transform,
            "mean_deformation_m": mean_deformation,
            "max_deformation_m": max_deformation,
            "rigid_translation_m": translation,
            "n_correspondences": n_common,
            "cumulative_max_deformation_m": self._cumulative_deformation_m,
            "frame_count": self._frame_count,
        }


# =============================================================================
# SECTION 4: DEPTH-TO-POINTCLOUD CONVERSION
# =============================================================================
# Convert depth images to 3D point clouds for registration and tracking.
# =============================================================================


class DepthToPointCloud:
    """
    Convert depth images to 3D point clouds using camera intrinsics.

    INSTRUCTIONS:
    - camera_matrix must match the depth camera's intrinsic calibration.
    - For RealSense D405: use rs2_intrinsics from the depth stream.
    - For stereo-derived depth: use the Q matrix from stereoRectify.
    - Set max_depth_m to reject far-field noise (typically 0.5 m for
      endoscopic view, 1.5 m for external camera).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_depth_m: float = 0.5,
        min_depth_m: float = 0.01,
    ):
        self.camera_matrix = camera_matrix
        self.max_depth_m = max_depth_m
        self.min_depth_m = min_depth_m
        self._fx = camera_matrix[0, 0]
        self._fy = camera_matrix[1, 1]
        self._cx = camera_matrix[0, 2]
        self._cy = camera_matrix[1, 2]

    def convert(self, depth_image: DepthImage) -> PointCloud:
        """
        Convert depth image to point cloud.

        Args:
            depth_image: DepthImage with depth_m and optional rgb.

        Returns:
            PointCloud in camera frame.
        """
        depth = depth_image.depth_m
        h, w = depth.shape

        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Filter valid depths
        valid = (depth > self.min_depth_m) & (depth < self.max_depth_m)
        u_valid = u[valid].astype(np.float32)
        v_valid = v[valid].astype(np.float32)
        z = depth[valid].astype(np.float32)

        # Back-project to 3D
        x = (u_valid - self._cx) * z / self._fx
        y = (v_valid - self._cy) * z / self._fy

        points = np.stack([x, y, z], axis=-1)

        # Extract colors if available
        colors = None
        if depth_image.rgb is not None:
            colors = depth_image.rgb[valid]

        return PointCloud(
            points_m=points,
            colors=colors,
            frame_id="camera",
            timestamp_ns=depth_image.timestamp_ns,
        )


# =============================================================================
# SECTION 5: TEMPORAL SYNCHRONIZATION
# =============================================================================
# Synchronize data from multiple sensors running at different rates.
# Critical for accurate fusion.
#
# INSTRUCTIONS:
# - All sensor timestamps must use the same clock source.
# - In ROS 2, use message_filters::ApproximateTimeSynchronizer.
# - Tolerance: typically 10 ms for visual+force fusion.
# =============================================================================


class TemporalSynchronizer:
    """
    Synchronize multi-sensor data streams by timestamp.

    INSTRUCTIONS:
    - Set tolerance_ns based on your sensors' relative timing uncertainty.
    - For hardware-synchronized stereo cameras: 1 ms tolerance.
    - For stereo + force-torque: 5 ms tolerance.
    - For stereo + optical tracker: 10 ms tolerance.

    The synchronizer buffers recent messages from each stream and
    emits synchronized bundles when all streams have data within
    the tolerance window.

    Example:
        >>> sync = TemporalSynchronizer(tolerance_ns=5_000_000)
        >>> sync.add_stream("camera", buffer_size=5)
        >>> sync.add_stream("force", buffer_size=20)
        >>> sync.push("camera", timestamp_ns, camera_data)
        >>> sync.push("force", timestamp_ns, force_data)
        >>> bundle = sync.get_synchronized()
    """

    def __init__(self, tolerance_ns: int = 10_000_000):
        self.tolerance_ns = tolerance_ns
        self._buffers: dict[str, list[tuple[int, Any]]] = {}
        self._buffer_sizes: dict[str, int] = {}

    def add_stream(self, name: str, buffer_size: int = 10):
        """Add a sensor stream to synchronize."""
        self._buffers[name] = []
        self._buffer_sizes[name] = buffer_size

    def push(self, stream_name: str, timestamp_ns: int, data: Any):
        """Push new data into a stream's buffer."""
        if stream_name not in self._buffers:
            return

        self._buffers[stream_name].append((timestamp_ns, data))

        # Trim old entries
        max_size = self._buffer_sizes[stream_name]
        if len(self._buffers[stream_name]) > max_size:
            self._buffers[stream_name] = self._buffers[stream_name][-max_size:]

    def get_synchronized(self) -> dict[str, Any] | None:
        """
        Try to extract a synchronized data bundle.

        Returns:
            Dictionary {stream_name: data} if all streams have data
            within tolerance, or None if synchronization fails.
        """
        if not self._buffers:
            return None

        # Check all streams have data
        for name, buf in self._buffers.items():
            if not buf:
                return None

        # Find the most recent timestamp from the slowest stream
        latest_per_stream = {
            name: buf[-1][0] for name, buf in self._buffers.items()
        }
        reference_time = min(latest_per_stream.values())

        # Find closest sample in each stream to reference time
        result = {}
        for name, buf in self._buffers.items():
            best_idx = -1
            best_diff = float("inf")
            for i, (ts, _data) in enumerate(buf):
                diff = abs(ts - reference_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_diff > self.tolerance_ns:
                return None  # This stream is too far from reference

            result[name] = buf[best_idx][1]

        return result


# =============================================================================
# SECTION 6: FUSED SCENE REPRESENTATION
# =============================================================================
# The fusion pipeline output: a unified scene that the robot controller
# and surgeon overlay system can consume.
# =============================================================================


@dataclass
class IntraoperativeScene:
    """
    Unified scene representation combining all sensor data.

    This is the output of the fusion pipeline, consumed by:
    - Robot controller (for autonomous positioning)
    - Surgeon display (for overlay visualization)
    - Safety monitor (for collision avoidance with anatomy)
    - Logging system (for post-operative analysis)
    """

    timestamp_ns: int

    # Instrument state
    instrument_tip_position_m: np.ndarray | None = None
    instrument_tip_orientation_quat: np.ndarray | None = None
    instrument_visible: bool = False

    # Tissue state
    tissue_point_cloud: PointCloud | None = None
    tissue_deformation_m: float = 0.0
    tissue_deformation_field: np.ndarray | None = None

    # Tumor margin estimate
    tumor_margin_distance_m: float | None = None
    margin_confidence: float = 0.0

    # Force at contact
    contact_force_n: float = 0.0
    contact_location_m: np.ndarray | None = None

    # Registration quality
    registration_error_mm: float = 0.0

    def is_margin_safe(self, min_margin_m: float = 0.005) -> bool:
        """Check if current margin distance is safe (>5 mm default)."""
        if self.tumor_margin_distance_m is None:
            return False
        return self.tumor_margin_distance_m >= min_margin_m


class SensorFusionPipeline:
    """
    Main sensor fusion pipeline for intraoperative perception.

    Combines data from all sensors into an IntraoperativeScene at
    the output rate (typically 15-30 Hz, limited by camera frame rate).

    ARCHITECTURE:
    =============
    Stereo Camera (30 Hz) ─┐
    Depth Camera (30 Hz)  ──┤
    Force-Torque (1 kHz)  ──┼── TemporalSync ── Fusion ── IntraoperativeScene
    Joint Encoders (1 kHz) ─┤
    Optical Tracker (60 Hz) ┘

    INSTRUCTIONS:
    - Call process() whenever new camera data arrives.
    - Force/torque and joint data are buffered and interpolated.
    - Output scene is published to ROS 2 topic or consumed directly.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        robot_to_camera_transform: np.ndarray | None = None,
    ):
        """
        Args:
            camera_matrix: 3x3 camera intrinsic matrix.
            robot_to_camera_transform: 4x4 transform from robot base
                to camera frame (from hand-eye calibration, example 04).
        """
        if robot_to_camera_transform is None:
            robot_to_camera_transform = np.eye(4)

        self.camera_matrix = camera_matrix
        self.T_robot_to_camera = robot_to_camera_transform

        # Processing components
        self._instrument_seg = InstrumentSegmenter()
        self._depth_to_pc = DepthToPointCloud(camera_matrix)
        self._tissue_tracker = TissueDeformationTracker()
        self._synchronizer = TemporalSynchronizer(tolerance_ns=15_000_000)

        # Setup synchronizer streams
        self._synchronizer.add_stream("depth", buffer_size=3)
        self._synchronizer.add_stream("force", buffer_size=50)

        self._frame_count = 0
        self._reference_set = False

        logger.info("SensorFusionPipeline initialized")

    def process(
        self,
        depth_image: DepthImage,
        force_xyz_n: np.ndarray,
        instrument_ee_position_m: np.ndarray,
        timestamp_ns: int,
    ) -> IntraoperativeScene:
        """
        Run one fusion cycle.

        Call this at camera frame rate (~30 Hz). Force and encoder data
        at higher rates should be pre-buffered via push_force().

        Args:
            depth_image: Current RGBD frame.
            force_xyz_n: Latest force reading [Fx, Fy, Fz] in Newtons.
            instrument_ee_position_m: End-effector position from robot FK.
            timestamp_ns: Current timestamp.

        Returns:
            IntraoperativeScene with fused data.
        """
        self._frame_count += 1
        scene = IntraoperativeScene(timestamp_ns=timestamp_ns)

        # --- Step 1: Instrument segmentation ---
        if depth_image.rgb is not None:
            mask = self._instrument_seg.segment(depth_image.rgb)
            tip_px = self._instrument_seg.get_tip_position_pixels(mask)
            if tip_px is not None:
                scene.instrument_visible = True
                # Use robot FK for 3D position (more accurate than vision)
                scene.instrument_tip_position_m = instrument_ee_position_m

        # --- Step 2: Depth to point cloud ---
        cloud = self._depth_to_pc.convert(depth_image)
        scene.tissue_point_cloud = cloud

        # --- Step 3: Tissue deformation tracking ---
        if not self._reference_set and cloud.points_m.shape[0] > 100:
            self._tissue_tracker.set_reference(cloud)
            self._reference_set = True
            logger.info("Reference point cloud set at frame %d", self._frame_count)

        if self._reference_set:
            deformation = self._tissue_tracker.track(cloud)
            if deformation.get("is_valid"):
                scene.tissue_deformation_m = deformation["max_deformation_m"]

        # --- Step 4: Contact force ---
        scene.contact_force_n = float(np.linalg.norm(force_xyz_n))
        if scene.contact_force_n > 0.1:
            scene.contact_location_m = instrument_ee_position_m

        # --- Step 5: Tumor margin estimate ---
        # In production, this uses the registered preop tumor segmentation
        # plus the current deformation field to compute distance from
        # instrument tip to nearest tumor boundary.
        scene.tumor_margin_distance_m = self._estimate_margin(
            instrument_ee_position_m
        )
        scene.margin_confidence = 0.85

        return scene

    def _estimate_margin(self, tip_position_m: np.ndarray) -> float:
        """
        Estimate distance from instrument tip to tumor margin.

        In production, this computes the distance from the tip position
        to the nearest point on the tumor boundary surface, accounting
        for tissue deformation since the preoperative scan.

        The tumor boundary comes from:
        1. Preoperative CT/MRI segmentation
        2. Registered to robot frame via patient registration
        3. Deformed using the tracked tissue deformation field
        """
        # Simulated: distance from tip to a hypothetical tumor boundary
        tumor_center = np.array([0.0, 0.0, -0.15])
        tumor_radius = 0.02
        dist_to_center = float(np.linalg.norm(tip_position_m - tumor_center))
        margin = dist_to_center - tumor_radius
        return max(0.0, margin)


# =============================================================================
# SECTION 7: MAIN DEMONSTRATION
# =============================================================================


def run_sensor_fusion_demo():
    """
    Demonstrate the multi-sensor fusion pipeline.

    Simulates 30 frames of a surgical procedure where the instrument
    approaches tissue and the system tracks deformation and margins.

    WHAT TO MODIFY FOR YOUR SYSTEM:
    - Replace simulated sensor data with actual hardware reads.
    - Adjust camera_matrix to match your calibration.
    - Set robot_to_camera_transform from hand-eye calibration (example 04).
    """
    logger.info("=" * 70)
    logger.info("MULTI-SENSOR FUSION FOR INTRAOPERATIVE PERCEPTION")
    logger.info("=" * 70)

    # Camera intrinsics (typical endoscope)
    camera_matrix = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )

    pipeline = SensorFusionPipeline(camera_matrix=camera_matrix)

    # Simulate 30 frames at 30 Hz
    n_frames = 30
    dt_s = 1.0 / 30.0
    margin_history = []

    for frame in range(n_frames):
        t_ns = int(frame * dt_s * 1e9)
        progress = frame / n_frames

        # Simulate instrument approaching tissue
        ee_pos = np.array([0.0, 0.0, -0.15 + progress * 0.02])
        force = np.array([0.0, 0.0, max(0.0, progress - 0.5) * 3.0])

        # Simulate depth image with tissue surface
        h, w = 480, 640
        depth = np.full((h, w), 0.20, dtype=np.float32)
        # Add tissue surface closer to camera in center
        depth[200:280, 280:360] = 0.15 - progress * 0.005
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Reddish tissue
        rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(int) + 80, 0, 255).astype(
            np.uint8
        )

        depth_img = DepthImage(
            depth_m=depth,
            rgb=rgb,
            timestamp_ns=t_ns,
            camera_matrix=camera_matrix,
        )

        # Run fusion
        scene = pipeline.process(
            depth_image=depth_img,
            force_xyz_n=force,
            instrument_ee_position_m=ee_pos,
            timestamp_ns=t_ns,
        )

        margin_history.append(scene.tumor_margin_distance_m)

        if frame % 10 == 0:
            margin_str = (
                f"{scene.tumor_margin_distance_m * 1000:.1f} mm"
                if scene.tumor_margin_distance_m is not None
                else "N/A"
            )
            logger.info(
                "Frame %d: instrument_visible=%s, deformation=%.2f mm, "
                "margin=%s, force=%.2f N",
                frame,
                scene.instrument_visible,
                scene.tissue_deformation_m * 1000,
                margin_str,
                scene.contact_force_n,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("SENSOR FUSION RESULTS")
    print("=" * 60)
    print(f"Frames processed:      {n_frames}")
    print(f"Final margin:          {margin_history[-1] * 1000:.1f} mm")
    print(
        f"Minimum margin:        "
        f"{min(m for m in margin_history if m is not None) * 1000:.1f} mm"
    )
    margin_safe = all(
        m is not None and m >= 0.005 for m in margin_history
    )
    print(f"Margin always safe:    {margin_safe}")

    return {
        "frames_processed": n_frames,
        "final_margin_mm": margin_history[-1] * 1000
        if margin_history[-1]
        else None,
        "margin_safe": margin_safe,
    }


if __name__ == "__main__":
    run_sensor_fusion_demo()
