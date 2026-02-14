"""Repository Module Architecture Treemap visualization.

Shows the full repository module hierarchy with relative code volume
so engineers understand where the bulk of the implementation lives.
"""

import os

import plotly.graph_objects as go


# --- Verified line counts from repository source files ---
MODULES = {
    "federation": {
        "federated_coordinator.py": 612,
        "differential_privacy.py": 490,
        "secure_aggregation.py": 556,
        "site_enrollment.py": 631,
        "data_harmonization.py": 678,
        "consortium_reporting.py": 591,
        "privacy_analytics.py": 630,
    },
    "regulatory-submit": {
        "presub_generator.py": 817,
        "pccp_engine.py": 706,
        "iec62304_generator.py": 891,
        "classification_advisor.py": 752,
        "clinical_evidence.py": 729,
        "audit_trail.py": 728,
    },
    "tools": {
        "deployment_readiness.py": 888,
        "dose_calculator.py": 548,
        "trial_site_monitor.py": 518,
        "dicom_inspector.py": 521,
        "sim_job_runner.py": 572,
    },
    "digital-twins": {
        "tumor_twin_pipeline.py": 907,
        "treatment_simulator.py": 729,
        "clinical_dt_interface.py": 1076,
    },
    "unification": {
        "isaac_mujoco_bridge.py": 556,
        "urdf_sdf_mjcf_converter.py": 663,
        "unified_agent_interface.py": 619,
        "framework_detector.py": 401,
        "validation_suite.py": 605,
    },
    "privacy": {
        "phi_detector.py": 677,
        "deidentification_pipeline.py": 810,
        "access_control_manager.py": 787,
        "breach_response_protocol.py": 709,
        "dua_generator.py": 649,
    },
    "regulatory": {
        "fda_submission_tracker.py": 656,
        "irb_protocol_manager.py": 642,
        "gcp_compliance_checker.py": 560,
        "regulatory_tracker.py": 509,
    },
    "q1-2026-standards": {
        "isaac_to_mujoco_pipeline.py": 831,
        "mujoco_to_isaac_pipeline.py": 722,
        "model_validator.py": 828,
        "benchmark_runner.py": 786,
    },
    "examples": {
        "surgical_robot_training.py": 1014,
        "digital_twin_surgical_planning.py": 705,
        "cross_framework_validation.py": 707,
        "agentic_clinical_workflow.py": 748,
        "treatment_response_prediction.py": 796,
    },
    "examples-new": {
        "realtime_safety_monitoring.py": 1009,
        "sensor_fusion_intraoperative.py": 877,
        "ros2_surgical_deployment.py": 1021,
        "hand_eye_calibration.py": 953,
        "shared_autonomy_teleop.py": 921,
        "robotic_sample_handling.py": 978,
    },
    "agentic-ai": {
        "mcp_clinical_robotics_server.py": 1080,
        "react_procedure_planner.py": 963,
        "realtime_adaptive_treatment.py": 975,
        "autonomous_sim_orchestrator.py": 961,
        "safety_constrained_executor.py": 886,
        "protocol_rag_compliance.py": 1124,
    },
    "tests": {
        "test_suite (all modules)": 14973,
    },
}

ROOT = "physical-ai-oncology-trials"


def build_treemap_data():
    """Build parallel lists for Plotly Treemap."""
    ids = [ROOT]
    labels = [ROOT]
    parents = [""]
    values = [0]

    for directory, files in MODULES.items():
        dir_id = f"{ROOT}/{directory}"
        ids.append(dir_id)
        labels.append(directory)
        parents.append(ROOT)
        values.append(0)

        for filename, lines in files.items():
            file_id = f"{dir_id}/{filename}"
            ids.append(file_id)
            labels.append(f"{filename}<br>{lines} lines")
            parents.append(dir_id)
            values.append(lines)

    return ids, labels, parents, values


def create_treemap(dark_mode=False):
    """Create repository architecture treemap figure."""
    ids, labels, parents, values = build_treemap_data()

    if dark_mode:
        template = "plotly_dark"
        colorscale = "Cividis"
        font_color = "#e0e0e0"
        bg_color = "#1a1a2e"
    else:
        template = "plotly_white"
        colorscale = "Teal"
        font_color = "#1a1a2e"
        bg_color = "#ffffff"

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colorscale=colorscale,
                cornerradius=3,
            ),
            textinfo="label+value",
            textfont=dict(size=11),
            pathbar=dict(textfont=dict(size=13)),
        )
    )

    fig.update_layout(
        title=dict(
            text="Physical AI Oncology Trials — Repository Module Architecture (v1.0.0)",
            font=dict(size=20, color=font_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        margin=dict(t=80, l=10, r=10, b=10),
        width=1920,
        height=1080,
    )

    return fig


def main():
    """Generate treemap visualizations in light and dark modes."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "png", "1st")
    os.makedirs(png_dir, exist_ok=True)

    for dark_mode in [False, True]:
        suffix = "dark" if dark_mode else "light"
        fig = create_treemap(dark_mode=dark_mode)

        html_path = os.path.join(script_dir, f"repo_architecture_treemap_{suffix}.html")
        fig.write_html(html_path, include_plotlyjs=True)

        png_path = os.path.join(png_dir, f"repo_architecture_treemap_{suffix}.png")
        fig.write_image(png_path, width=1920, height=1080, scale=2)

    print("repo_architecture_treemap: done")


if __name__ == "__main__":
    main()
