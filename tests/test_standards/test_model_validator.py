"""Tests for Robot Model Validator.

Validates ValidationResult, ModelValidationReport, FormatValidator,
KinematicValidator, DynamicValidator, CrossFrameworkValidator, and
the ModelValidator orchestrator.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "q1-2026-standards/objective-2-robot-model-repository/model_validator.py"


@pytest.fixture()
def mv_mod():
    return load_module("model_validator", MOD_PATH)


def _make_urdf(tmp_path, name="val_robot", good=True):
    """Create a URDF file for testing."""
    effort = "50" if good else "0"
    velocity = "2.0" if good else "0"
    lower = "-1.57"
    upper = "1.57" if good else "-2.0"
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>
  </joint>
</robot>"""
    path = tmp_path / f"{name}.urdf"
    path.write_text(urdf)
    return str(path)


def _make_mjcf(tmp_path, name="val_robot"):
    """Create a minimal MJCF file for testing."""
    mjcf = f"""<?xml version="1.0"?>
<mujoco model="{name}">
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="link1" pos="0 0 0.4">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    path = tmp_path / f"{name}.xml"
    path.write_text(mjcf)
    return str(path)


def _make_sdf(tmp_path, name="val_robot"):
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <link name="base"/>
  </model>
</sdf>"""
    path = tmp_path / f"{name}.sdf"
    path.write_text(sdf)
    return str(path)


# ===========================================================================
# TestValidationResult
# ===========================================================================


class TestValidationResult:
    def test_construction(self, mv_mod):
        r = mv_mod.ValidationResult(
            check_name="test",
            passed=True,
            level=1,
            message="OK",
        )
        assert r.passed is True
        assert r.level == 1
        assert r.details == {}

    def test_with_details(self, mv_mod):
        r = mv_mod.ValidationResult(
            check_name="mass",
            passed=False,
            level=2,
            message="Invalid mass",
            details={"total_mass": -1},
        )
        assert r.details["total_mass"] == -1


# ===========================================================================
# TestModelValidationReport
# ===========================================================================


class TestModelValidationReport:
    def test_initial_state(self, mv_mod):
        report = mv_mod.ModelValidationReport(
            model_path="/tmp/robot.urdf",
            model_name="robot",
            validation_level=4,
        )
        assert report.overall_passed is True
        assert report.results == []

    def test_add_result_passing(self, mv_mod):
        report = mv_mod.ModelValidationReport(
            model_path="/tmp/robot.urdf",
            model_name="robot",
            validation_level=4,
        )
        report.add_result(mv_mod.ValidationResult("t", True, 1, "OK"))
        assert report.overall_passed is True

    def test_add_result_failing(self, mv_mod):
        report = mv_mod.ModelValidationReport(
            model_path="/tmp/robot.urdf",
            model_name="robot",
            validation_level=4,
        )
        report.add_result(mv_mod.ValidationResult("t", False, 1, "FAIL"))
        assert report.overall_passed is False

    def test_to_dict(self, mv_mod):
        report = mv_mod.ModelValidationReport(
            model_path="/tmp/robot.urdf",
            model_name="robot",
            validation_level=2,
        )
        report.add_result(mv_mod.ValidationResult("check1", True, 1, "OK"))
        d = report.to_dict()
        assert d["model_name"] == "robot"
        assert d["validation_level"] == 2
        assert len(d["results"]) == 1
        assert "timestamp" in d


# ===========================================================================
# TestFormatValidator
# ===========================================================================


class TestFormatValidator:
    def test_validate_urdf_valid(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        fv = mv_mod.FormatValidator()
        result = fv.validate_urdf(urdf)
        assert result.passed is True
        assert result.level == 1
        assert result.details["links"] == 2

    def test_validate_urdf_missing_file(self, mv_mod):
        fv = mv_mod.FormatValidator()
        result = fv.validate_urdf("/nonexistent/robot.urdf")
        assert result.passed is False

    def test_validate_mjcf_valid(self, mv_mod, tmp_path):
        mjcf = _make_mjcf(tmp_path)
        fv = mv_mod.FormatValidator()
        result = fv.validate_mjcf(mjcf)
        assert result.passed is True
        assert result.details["bodies"] >= 1

    def test_validate_mjcf_missing_worldbody(self, mv_mod, tmp_path):
        bad_mjcf = tmp_path / "bad.xml"
        bad_mjcf.write_text('<?xml version="1.0"?><mujoco model="bad"></mujoco>')
        fv = mv_mod.FormatValidator()
        result = fv.validate_mjcf(str(bad_mjcf))
        assert result.passed is False
        assert "worldbody" in result.message.lower()

    def test_validate_sdf_valid(self, mv_mod, tmp_path):
        sdf = _make_sdf(tmp_path)
        fv = mv_mod.FormatValidator()
        result = fv.validate_sdf(sdf)
        assert result.passed is True

    def test_validate_sdf_missing_model(self, mv_mod, tmp_path):
        bad_sdf = tmp_path / "bad.sdf"
        bad_sdf.write_text('<?xml version="1.0"?><sdf version="1.9"></sdf>')
        fv = mv_mod.FormatValidator()
        result = fv.validate_sdf(str(bad_sdf))
        assert result.passed is False


# ===========================================================================
# TestKinematicValidator
# ===========================================================================


class TestKinematicValidator:
    def test_mass_properties_valid(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path, good=True)
        kv = mv_mod.KinematicValidator()
        result = kv.validate_mass_properties(urdf)
        assert result.passed is True
        assert result.details["total_mass"] == 7.0

    def test_joint_limits_valid(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path, good=True)
        kv = mv_mod.KinematicValidator()
        result = kv.validate_joint_limits(urdf)
        assert result.passed is True

    def test_joint_limits_invalid(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path, name="bad_robot", good=False)
        kv = mv_mod.KinematicValidator()
        result = kv.validate_joint_limits(urdf)
        assert result.passed is False

    def test_link_consistency_valid(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path, good=True)
        kv = mv_mod.KinematicValidator()
        result = kv.validate_link_consistency(urdf)
        assert result.passed is True

    def test_link_consistency_invalid_parent(self, mv_mod, tmp_path):
        bad_urdf = tmp_path / "bad_parent.urdf"
        bad_urdf.write_text("""<?xml version="1.0"?>
<robot name="bad">
  <link name="base"/>
  <joint name="j" type="fixed">
    <parent link="nonexistent"/>
    <child link="base"/>
  </joint>
</robot>""")
        kv = mv_mod.KinematicValidator()
        result = kv.validate_link_consistency(str(bad_urdf))
        assert result.passed is False


# ===========================================================================
# TestDynamicValidator
# ===========================================================================


class TestDynamicValidator:
    def test_stability_without_mujoco(self, mv_mod):
        dv = mv_mod.DynamicValidator()
        result = dv.validate_physics_stability("nonexistent.xml")
        # Without MuJoCo, should return passed=False
        assert isinstance(result, mv_mod.ValidationResult)
        assert result.level == 3

    def test_actuator_response_without_mujoco(self, mv_mod):
        dv = mv_mod.DynamicValidator()
        result = dv.validate_actuator_response("nonexistent.xml")
        assert isinstance(result, mv_mod.ValidationResult)
        assert result.level == 3


# ===========================================================================
# TestCrossFrameworkValidator
# ===========================================================================


class TestCrossFrameworkValidator:
    def test_format_consistency_matching(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        mjcf = _make_mjcf(tmp_path)
        cfv = mv_mod.CrossFrameworkValidator()
        result = cfv.validate_format_consistency(urdf, mjcf)
        assert isinstance(result, mv_mod.ValidationResult)
        assert result.level == 4

    def test_format_consistency_missing_file(self, mv_mod):
        cfv = mv_mod.CrossFrameworkValidator()
        result = cfv.validate_format_consistency("no.urdf", "no.xml")
        assert result.passed is False


# ===========================================================================
# TestModelValidator
# ===========================================================================


class TestModelValidator:
    def test_validate_urdf_level_1(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        validator = mv_mod.ModelValidator()
        report = validator.validate_model(urdf, level=1)
        assert isinstance(report, mv_mod.ModelValidationReport)
        assert len(report.results) >= 1

    def test_validate_urdf_level_2(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        validator = mv_mod.ModelValidator()
        report = validator.validate_model(urdf, level=2)
        level2_results = [r for r in report.results if r.level == 2]
        assert len(level2_results) >= 2

    def test_validate_mjcf_level_1(self, mv_mod, tmp_path):
        mjcf = _make_mjcf(tmp_path)
        validator = mv_mod.ModelValidator()
        report = validator.validate_model(mjcf, level=1)
        assert len(report.results) >= 1

    def test_model_name_from_file(self, mv_mod, tmp_path):
        urdf = _make_urdf(tmp_path)
        validator = mv_mod.ModelValidator()
        report = validator.validate_model(urdf, level=1)
        assert report.model_name == "val_robot"
