"""Configuration constants for the 168-hour sponsor simulation."""

from __future__ import annotations

# 7-day simulation structure
TOTAL_DAYS = 7
HOURS_PER_DAY = 24
TOTAL_HOURS = TOTAL_DAYS * HOURS_PER_DAY  # 168

DAY_THEMES = {
    1: "Trial Initialization and Baseline Operations",
    2: "Enrollment Acceleration and Protocol Optimization",
    3: "Mid-Trial Safety Review and Adaptive Modifications",
    4: "Robotic Fleet Scaling and Cross-Site Coordination",
    5: "Data Analysis and Interim Reporting",
    6: "Regulatory Compliance and Audit Preparation",
    7: "Trial Closeout and Final Documentation",
}

# Period labels for each hour-of-day (0-23)
PERIOD_LABELS = [
    "overnight_low",
    "overnight_low",
    "overnight_low",
    "early_morning",
    "early_morning",
    "early_morning",
    "morning_ramp",
    "morning_peak",
    "morning_peak",
    "peak_operations",
    "peak_operations",
    "peak_operations",
    "midday_steady",
    "afternoon_ops",
    "afternoon_ops",
    "afternoon_steady",
    "afternoon_wind",
    "evening_transition",
    "evening_ops",
    "evening_wind",
    "night_transition",
    "night_ops",
    "overnight_transition",
    "overnight_transition",
]

PERIOD_DESCRIPTIONS = {
    "overnight_low": "Overnight low-volume operations with minimal staffing",
    "early_morning": "Early morning pre-dawn operations with gradual ramp-up",
    "morning_ramp": "Morning ramp-up with increasing patient arrivals",
    "morning_peak": "Morning peak operations with full staffing",
    "peak_operations": "Peak operational period with maximum throughput",
    "midday_steady": "Midday steady-state operations",
    "afternoon_ops": "Afternoon clinical operations",
    "afternoon_steady": "Afternoon steady-state with moderate volume",
    "afternoon_wind": "Late afternoon wind-down beginning",
    "evening_transition": "Evening transition to reduced operations",
    "evening_ops": "Evening operations with reduced staffing",
    "evening_wind": "Evening wind-down period",
    "night_transition": "Night transition to minimal operations",
    "night_ops": "Night operations with skeleton crew",
    "overnight_transition": "Overnight transition to minimal operations",
}

# Patient volumes per hour-of-day for each day (7 x 24 matrix)
PATIENT_VOLUMES = {
    1: [2, 3, 3, 4, 4, 5, 8, 10, 12, 15, 12, 10, 8, 9, 10, 8, 7, 6, 5, 4, 3, 3, 2, 2],
    2: [3, 4, 4, 5, 5, 7, 10, 13, 15, 18, 15, 12, 10, 11, 12, 10, 9, 8, 6, 5, 4, 4, 3, 3],
    3: [3, 4, 4, 5, 6, 8, 12, 14, 16, 20, 16, 14, 11, 12, 13, 11, 10, 8, 7, 6, 4, 4, 3, 3],
    4: [4, 5, 5, 6, 6, 8, 11, 14, 17, 21, 17, 14, 12, 13, 14, 12, 10, 9, 7, 6, 5, 4, 3, 3],
    5: [3, 4, 5, 5, 6, 7, 10, 13, 15, 18, 15, 13, 10, 11, 12, 10, 9, 7, 6, 5, 4, 3, 3, 2],
    6: [3, 4, 4, 5, 5, 7, 9, 12, 14, 17, 14, 12, 9, 10, 11, 9, 8, 7, 5, 4, 4, 3, 2, 2],
    7: [2, 3, 3, 4, 4, 6, 8, 10, 12, 14, 12, 10, 8, 9, 10, 8, 7, 5, 4, 3, 3, 2, 2, 1],
}

# Escalation counts per hour-of-day for each day
ESCALATION_COUNTS = {
    1: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 3, 2, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# PSL score ranges per day (start, end)
PSL_RANGES = {
    1: (63.4, 64.8),
    2: (64.8, 66.0),
    3: (66.0, 67.0),
    4: (67.0, 68.0),
    5: (68.0, 68.8),
    6: (68.8, 69.5),
    7: (69.5, 70.0),
}

AGENTS = [
    "portfolio_agent",
    "asset_lead_agent",
    "clinical_accountability_agent",
    "study_orchestrator",
    "clinops_agent",
    "safety_agent",
    "regulatory_agent",
    "quality_agent",
    "supply_agent",
    "data_biostats_agent",
    "site_gateway",
    "robot_execution_gateway",
]

DECISION_TYPES = [
    "INIT",
    "ENROLL",
    "MONITOR",
    "AUTH",
    "SAFETY_CHECK",
    "SUPPLY",
    "DATA_QUALITY",
    "ESCALATION",
    "STATUS",
    "PROCEDURE",
    "DISCHARGE",
    "REGULATORY",
]

ROBOT_FLEET = [
    {"category": "Surgical Robots (da Vinci Xi)", "instances": 3},
    {"category": "Collaborative Robots (Cobots, Franka Emika)", "instances": 4},
    {"category": "RT Positioning Robots", "instances": 3},
    {"category": "Needle Placement Systems", "instances": 2},
    {"category": "Social Companion Robots", "instances": 5},
    {"category": "Rehabilitation Exoskeletons", "instances": 2},
    {"category": "RT Motion-Tracking Systems", "instances": 3},
    {"category": "Imaging Assistance Robots", "instances": 4},
    {"category": "Transport/Logistics Robots", "instances": 2},
    {"category": "Environmental Monitoring Robots", "instances": 1},
]

ROBOT_CATEGORIES_SHORT = [
    "IV-Admin",
    "Vitals-Mon",
    "Lab-Draw",
    "Imaging-Pos",
    "Med-Dispense",
    "Specimen-Xfer",
]

GATE_LABELS = {"G1": "Auto", "G2": "Clinician", "G3": "ParamValid", "G4": "MandHuman"}

CANCER_TYPES = [
    "NSCLC adenocarcinoma",
    "HCC",
    "Breast IDC",
    "Prostate adenocarcinoma",
    "Pediatric ALL",
    "Glioblastoma",
    "Colorectal adenocarcinoma",
    "Melanoma",
    "Ovarian serous",
    "Pancreatic ductal",
    "Renal cell carcinoma",
    "Bladder urothelial",
    "Esophageal SCC",
    "Thyroid papillary",
    "Endometrial",
    "Gastric adenocarcinoma",
    "Head and neck SCC",
    "Cervical SCC",
    "Multiple myeloma",
    "Non-Hodgkin lymphoma",
    "CML",
    "AML",
    "Cholangiocarcinoma",
    "Mesothelioma",
    "Sarcoma",
    "Testicular seminoma",
    "Wilms tumor",
    "Neuroblastoma",
    "Medulloblastoma",
    "Retinoblastoma",
]

STAGES = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV", "HR"]

ROBOT_ASSIGNMENT_CATEGORIES = [
    "Surgical (da Vinci Xi)",
    "RT Motion-Tracking",
    "Imaging Assistance",
    "Needle Placement",
    "Social Companion",
    "Rehabilitation Exoskeleton",
    "RT Positioning",
    "Transport/Logistics",
    "Cobots (Franka Emika)",
    "Environmental Monitoring",
]

# Base date for timestamps - Day 1 starts 2026-03-23
BASE_DATE = "2026-03-23"
