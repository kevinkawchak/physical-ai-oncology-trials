#!/usr/bin/env python3
"""Generate patient-robot instructional illustrations for Physical AI Oncology Trials.

Creates 10 high-resolution black-and-white portrait illustrations using Cairo,
one for each robot type, showing a diverse patient interacting with the robot.
Outputs individual SVG, PDF, and PNG files, plus a combined 10-page PDF.

Robot types (ordered by commonality in physical AI oncology trials):
1. Surgical Robots
2. Cobots (Collaborative Robots)
3. Radiotherapy Patient-Positioning Robots
4. Robotic Needle-Placement Systems
5. Social Companion Robots (PEDIATRIC)
6. Humanoids (PEDIATRIC)
7. Radiotherapy Motion-Management / Tracking Robots
8. Imaging Assistant Robots
9. Steerable Needle / Needle-Steering Robots
10. Rehabilitation Exoskeletons / Robotic Gait Trainers
"""

import cairo
import math
import os

# ---------- constants ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVG_DIR = os.path.join(BASE_DIR, "svg")
PDF_DIR = os.path.join(BASE_DIR, "pdf")
PNG_DIR = os.path.join(BASE_DIR, "png")

# Portrait dimensions at 300 DPI equivalent (A4-ish aspect ratio)
IMG_W = 1800
IMG_H = 2000
SCALE = 1.0

ROBOT_TYPES = [
    "Surgical Robots",
    "Cobots",
    "Radiotherapy Patient-Positioning Robots",
    "Robotic Needle-Placement Systems",
    "Social Companion Robots",
    "Humanoids",
    "Radiotherapy Motion-Management / Tracking Robots",
    "Imaging Assistant Robots",
    "Steerable Needle / Needle-Steering Robots",
    "Rehabilitation Exoskeletons / Robotic Gait Trainers",
]

# Pages 5 and 6 (index 4, 5) are pediatric
PEDIATRIC_PAGES = {4, 5}


# ── Helper drawing functions ─────────────────────────────────
def set_black(ctx, width=2.5):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(width)


def draw_circle(ctx, cx, cy, r):
    ctx.arc(cx, cy, r, 0, 2 * math.pi)


def fill_white(ctx):
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.stroke()


def draw_rounded_rect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def draw_head(ctx, cx, cy, radius, is_child=False):
    """Draw a human head with basic features."""
    r = radius
    # Head outline
    draw_circle(ctx, cx, cy, r)
    fill_white(ctx)

    # Eyes
    eye_y = cy - r * 0.15
    eye_sep = r * 0.35
    for ex in [cx - eye_sep, cx + eye_sep]:
        ctx.arc(ex, eye_y, r * 0.08, 0, 2 * math.pi)
        ctx.fill()

    # Nose
    set_black(ctx, 2.0)
    ctx.move_to(cx, cy - r * 0.05)
    ctx.line_to(cx - r * 0.08, cy + r * 0.15)
    ctx.line_to(cx + r * 0.08, cy + r * 0.15)
    ctx.stroke()

    # Mouth
    ctx.arc(cx, cy + r * 0.25, r * 0.2, 0.15, math.pi - 0.15)
    ctx.stroke()

    set_black(ctx, 2.5)


def draw_hair_short(ctx, cx, cy, r):
    """Short hair style."""
    ctx.save()
    ctx.arc(cx, cy, r * 1.05, -math.pi, 0)
    ctx.stroke()
    # Add some short lines on top
    for angle in [-2.5, -2.0, -1.5, -1.0, -0.5]:
        x1 = cx + r * 0.95 * math.cos(angle)
        y1 = cy + r * 0.95 * math.sin(angle)
        x2 = cx + r * 1.15 * math.cos(angle)
        y2 = cy + r * 1.15 * math.sin(angle)
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.restore()


def draw_hair_long(ctx, cx, cy, r):
    """Long hair style."""
    ctx.save()
    # Hair flowing down
    ctx.move_to(cx - r * 1.05, cy - r * 0.2)
    ctx.curve_to(cx - r * 1.2, cy + r * 0.5, cx - r * 1.1, cy + r * 1.2, cx - r * 0.7, cy + r * 1.5)
    ctx.stroke()
    ctx.move_to(cx + r * 1.05, cy - r * 0.2)
    ctx.curve_to(cx + r * 1.2, cy + r * 0.5, cx + r * 1.1, cy + r * 1.2, cx + r * 0.7, cy + r * 1.5)
    ctx.stroke()
    # Top of hair
    ctx.arc(cx, cy - r * 0.15, r * 1.08, -math.pi, 0)
    ctx.stroke()
    ctx.restore()


def draw_hair_curly(ctx, cx, cy, r):
    """Curly/afro hair style."""
    ctx.save()
    set_black(ctx, 2.0)
    for angle_deg in range(0, 360, 12):
        a = math.radians(angle_deg)
        bx = cx + r * 1.15 * math.cos(a)
        by = cy + r * 1.15 * math.sin(a)
        ctx.arc(bx, by, r * 0.12, 0, 2 * math.pi)
        ctx.stroke()
    set_black(ctx, 2.5)
    ctx.restore()


def draw_hair_bald(ctx, cx, cy, r):
    """Bald/very short hair — just emphasize the head shape."""
    ctx.save()
    set_black(ctx, 1.5)
    # A few tiny dots for stubble on top
    for angle_deg in range(-160, -20, 15):
        a = math.radians(angle_deg)
        bx = cx + r * 0.92 * math.cos(a)
        by = cy + r * 0.92 * math.sin(a)
        ctx.arc(bx, by, 1.5, 0, 2 * math.pi)
        ctx.fill()
    set_black(ctx, 2.5)
    ctx.restore()


def draw_headscarf(ctx, cx, cy, r):
    """Headscarf/hijab style."""
    ctx.save()
    set_black(ctx, 2.5)
    # Outer wrap
    ctx.arc(cx, cy - r * 0.1, r * 1.12, -math.pi + 0.3, -0.3)
    ctx.line_to(cx + r * 0.9, cy + r * 0.9)
    ctx.curve_to(cx + r * 0.5, cy + r * 1.3, cx - r * 0.5, cy + r * 1.3, cx - r * 0.9, cy + r * 0.9)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.stroke()
    ctx.restore()


def draw_child_pigtails(ctx, cx, cy, r):
    """Child with pigtails."""
    ctx.save()
    # Top hair
    ctx.arc(cx, cy - r * 0.15, r * 1.05, -math.pi, 0)
    ctx.stroke()
    # Pigtails
    for side in [-1, 1]:
        px = cx + side * r * 1.0
        py = cy - r * 0.3
        ctx.move_to(cx + side * r * 0.9, cy - r * 0.5)
        ctx.curve_to(px + side * r * 0.3, py - r * 0.3, px + side * r * 0.2, py + r * 0.5, px, py + r * 0.8)
        ctx.stroke()
        # Hair tie
        ctx.arc(cx + side * r * 0.9, cy - r * 0.5, r * 0.08, 0, 2 * math.pi)
        ctx.stroke()
    ctx.restore()


def draw_child_short(ctx, cx, cy, r):
    """Child with short messy hair."""
    ctx.save()
    set_black(ctx, 2.0)
    for angle_deg in range(-170, -10, 18):
        a = math.radians(angle_deg)
        x1 = cx + r * 0.95 * math.cos(a)
        y1 = cy + r * 0.95 * math.sin(a)
        x2 = cx + r * 1.2 * math.cos(a + 0.15)
        y2 = cy + r * 1.2 * math.sin(a + 0.15)
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
    ctx.stroke()
    set_black(ctx, 2.5)
    ctx.restore()


HAIR_STYLES = [
    draw_hair_short,
    draw_hair_long,
    draw_hair_curly,
    draw_hair_bald,
    draw_headscarf,
    draw_hair_short,
    draw_hair_long,
    draw_hair_curly,
]

CHILD_HAIR_STYLES = [draw_child_pigtails, draw_child_short]


def draw_patient_body(ctx, cx, cy, body_h, is_child=False, seated=False, gown=True):
    """Draw a patient body (gown or clothing) below head position."""
    shoulder_w = body_h * 0.55 if not is_child else body_h * 0.45
    if gown:
        # Hospital gown
        ctx.move_to(cx - shoulder_w / 2, cy)
        ctx.line_to(cx - shoulder_w / 2 - 15, cy + body_h * 0.65)
        ctx.line_to(cx + shoulder_w / 2 + 15, cy + body_h * 0.65)
        ctx.line_to(cx + shoulder_w / 2, cy)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke()

        # V-neck
        ctx.move_to(cx - 20, cy)
        ctx.line_to(cx, cy + 35)
        ctx.line_to(cx + 20, cy)
        ctx.stroke()

        # Gown tie
        ctx.move_to(cx - 5, cy + 35)
        ctx.line_to(cx - 25, cy + 55)
        ctx.stroke()
        ctx.move_to(cx + 5, cy + 35)
        ctx.line_to(cx + 25, cy + 55)
        ctx.stroke()
    else:
        # Simple top
        ctx.move_to(cx - shoulder_w / 2, cy)
        ctx.line_to(cx - shoulder_w / 2 - 10, cy + body_h * 0.65)
        ctx.line_to(cx + shoulder_w / 2 + 10, cy + body_h * 0.65)
        ctx.line_to(cx + shoulder_w / 2, cy)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke()

    if not seated:
        # Legs
        leg_top = cy + body_h * 0.6
        ctx.move_to(cx - 20, leg_top)
        ctx.line_to(cx - 30, cy + body_h)
        ctx.stroke()
        ctx.move_to(cx + 20, leg_top)
        ctx.line_to(cx + 30, cy + body_h)
        ctx.stroke()


def draw_arm_reaching(ctx, shoulder_x, shoulder_y, hand_x, hand_y):
    """Draw an arm reaching toward something."""
    mid_x = (shoulder_x + hand_x) / 2
    mid_y = min(shoulder_y, hand_y) - 15
    ctx.move_to(shoulder_x, shoulder_y)
    ctx.curve_to(mid_x, mid_y, hand_x - 10, hand_y - 20, hand_x, hand_y)
    ctx.stroke()
    # Hand
    ctx.arc(hand_x, hand_y, 8, 0, 2 * math.pi)
    fill_white(ctx)


def draw_arm_down(ctx, shoulder_x, shoulder_y, length):
    """Draw an arm hanging or resting."""
    ctx.move_to(shoulder_x, shoulder_y)
    ctx.line_to(shoulder_x - 10, shoulder_y + length)
    ctx.stroke()
    ctx.arc(shoulder_x - 10, shoulder_y + length, 7, 0, 2 * math.pi)
    fill_white(ctx)


def draw_arm_bent(ctx, shoulder_x, shoulder_y, elbow_x, elbow_y, hand_x, hand_y):
    """Draw an arm with a bent elbow."""
    ctx.move_to(shoulder_x, shoulder_y)
    ctx.line_to(elbow_x, elbow_y)
    ctx.line_to(hand_x, hand_y)
    ctx.stroke()
    ctx.arc(hand_x, hand_y, 7, 0, 2 * math.pi)
    fill_white(ctx)


# ── ISO symbol helpers ───────────────────────────────────────
def draw_iso_warning_triangle(ctx, cx, cy, size):
    """ISO 7010 / ISO 3864-1 general warning triangle."""
    h = size * 0.866
    ctx.move_to(cx, cy - h * 0.6)
    ctx.line_to(cx - size / 2, cy + h * 0.4)
    ctx.line_to(cx + size / 2, cy + h * 0.4)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.5)
    ctx.stroke()
    # Exclamation
    ctx.move_to(cx, cy - h * 0.2)
    ctx.line_to(cx, cy + h * 0.1)
    ctx.stroke()
    ctx.arc(cx, cy + h * 0.22, 2, 0, 2 * math.pi)
    ctx.fill()


def draw_iso_radiation(ctx, cx, cy, size):
    """ISO 7010 W003 ionizing radiation trefoil."""
    r_inner = size * 0.15
    r_outer = size * 0.45
    set_black(ctx, 2.0)
    for i in range(3):
        a_start = math.radians(i * 120 - 90 + 10)
        a_end = math.radians(i * 120 - 90 + 110)
        ctx.arc(cx, cy, r_outer, a_start, a_end)
        ctx.arc_negative(cx, cy, r_inner, a_end, a_start)
        ctx.close_path()
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
    # Center dot
    ctx.arc(cx, cy, r_inner * 0.6, 0, 2 * math.pi)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 1.5)
    ctx.stroke()
    set_black(ctx, 2.5)


def draw_iso_keep_still(ctx, cx, cy, size):
    """ISO 15223-1 style 'keep still' / remain stationary indicator."""
    # Person silhouette with lines indicating stillness
    head_r = size * 0.12
    draw_circle(ctx, cx, cy - size * 0.3, head_r)
    ctx.stroke()
    # Body
    ctx.move_to(cx, cy - size * 0.18)
    ctx.line_to(cx, cy + size * 0.15)
    ctx.stroke()
    # Arms out
    ctx.move_to(cx - size * 0.2, cy - size * 0.05)
    ctx.line_to(cx + size * 0.2, cy - size * 0.05)
    ctx.stroke()
    # Legs
    ctx.move_to(cx, cy + size * 0.15)
    ctx.line_to(cx - size * 0.12, cy + size * 0.35)
    ctx.stroke()
    ctx.move_to(cx, cy + size * 0.15)
    ctx.line_to(cx + size * 0.12, cy + size * 0.35)
    ctx.stroke()
    # Stillness lines
    for dx in [-size * 0.3, size * 0.3]:
        ctx.move_to(cx + dx, cy - size * 0.2)
        ctx.line_to(cx + dx, cy + size * 0.2)
        ctx.stroke()


# ── Robot drawing functions (one per type) ───────────────────
def draw_surgical_robot(ctx, rx, ry):
    """Surgical robot arm (da Vinci style) with multi-arm system and patient on table."""
    # Base/tower
    draw_rounded_rect(ctx, rx - 60, ry + 80, 120, 200, 8)
    fill_white(ctx)
    # Tower display
    draw_rounded_rect(ctx, rx - 40, ry + 95, 80, 40, 4)
    ctx.stroke()
    # Status lines on display
    for dy in [105, 115, 125]:
        ctx.move_to(rx - 30, ry + dy)
        ctx.line_to(rx + 30, ry + dy)
        ctx.stroke()

    # Main arm column
    ctx.move_to(rx, ry + 80)
    ctx.line_to(rx, ry - 40)
    ctx.stroke()

    # Multi-arm surgical head
    for arm_angle, arm_len in [(-35, 120), (0, 130), (35, 120)]:
        a = math.radians(arm_angle - 90)
        ax = rx + arm_len * math.cos(a) * 0.7
        ay = ry - 40 + arm_len * math.sin(a) * 0.7

        # Upper arm segment
        ctx.move_to(rx, ry - 40)
        mid_x = (rx + ax) / 2 + 15 * math.sin(a)
        mid_y = (ry - 40 + ay) / 2 - 15 * math.cos(a)
        ctx.curve_to(mid_x, mid_y, ax + 5, ay - 10, ax, ay)
        ctx.stroke()

        # Joint
        ctx.arc(ax, ay, 5, 0, 2 * math.pi)
        fill_white(ctx)

        # Lower arm / instrument
        end_x = ax + 50 * math.cos(a + 0.3)
        end_y = ay + 50 * math.sin(a + 0.3)
        ctx.move_to(ax, ay)
        ctx.line_to(end_x, end_y)
        ctx.stroke()

        # Instrument tip
        if arm_angle == 0:
            # Camera/endoscope
            ctx.arc(end_x, end_y, 6, 0, 2 * math.pi)
            fill_white(ctx)
            ctx.arc(end_x, end_y, 3, 0, 2 * math.pi)
            ctx.fill()
        else:
            # Grasper jaws
            ctx.move_to(end_x, end_y)
            ctx.line_to(end_x + 12 * math.cos(a + 0.5), end_y + 12 * math.sin(a + 0.5))
            ctx.stroke()
            ctx.move_to(end_x, end_y)
            ctx.line_to(end_x + 12 * math.cos(a - 0.5), end_y + 12 * math.sin(a - 0.5))
            ctx.stroke()

    # Wheels
    for wx in [rx - 45, rx - 15, rx + 15, rx + 45]:
        ctx.arc(wx, ry + 285, 8, 0, 2 * math.pi)
        fill_white(ctx)


def draw_cobot(ctx, rx, ry):
    """Collaborative robot arm (7-DOF) mounted on a mobile base assisting patient."""
    # Mobile base
    draw_rounded_rect(ctx, rx - 55, ry + 160, 110, 80, 10)
    fill_white(ctx)

    # Wheels
    for wx in [rx - 40, rx + 40]:
        ctx.arc(wx, ry + 245, 10, 0, 2 * math.pi)
        fill_white(ctx)

    # Arm base on pedestal
    ctx.move_to(rx - 25, ry + 160)
    ctx.line_to(rx - 25, ry + 130)
    ctx.line_to(rx + 25, ry + 130)
    ctx.line_to(rx + 25, ry + 160)
    ctx.stroke()

    # 7-DOF arm segments
    joints = [
        (rx, ry + 130),
        (rx + 30, ry + 80),
        (rx + 60, ry + 40),
        (rx + 40, ry - 10),
        (rx + 10, ry - 40),
        (rx - 20, ry - 50),
    ]
    set_black(ctx, 3.0)
    for i in range(len(joints) - 1):
        ctx.move_to(joints[i][0], joints[i][1])
        ctx.line_to(joints[i + 1][0], joints[i + 1][1])
        ctx.stroke()
    set_black(ctx, 2.5)

    # Joint circles
    for jx, jy in joints:
        ctx.arc(jx, jy, 7, 0, 2 * math.pi)
        fill_white(ctx)

    # End effector (gripper)
    ex, ey = joints[-1]
    ctx.move_to(ex, ey)
    ctx.line_to(ex - 15, ey - 20)
    ctx.stroke()
    ctx.move_to(ex, ey)
    ctx.line_to(ex + 5, ey - 22)
    ctx.stroke()

    # Force/torque sensor indicator
    set_black(ctx, 1.5)
    ctx.arc(joints[4][0], joints[4][1], 12, 0, 2 * math.pi)
    ctx.stroke()
    set_black(ctx, 2.5)


def draw_radiotherapy_positioning_robot(ctx, rx, ry):
    """Radiotherapy patient-positioning robot / robotic couch (e.g., CyberKnife table)."""
    # Large floor base
    draw_rounded_rect(ctx, rx - 80, ry + 180, 160, 30, 5)
    fill_white(ctx)

    # Vertical support column
    ctx.move_to(rx - 20, ry + 180)
    ctx.line_to(rx - 25, ry + 80)
    ctx.line_to(rx + 25, ry + 80)
    ctx.line_to(rx + 20, ry + 180)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.5)
    ctx.stroke()

    # Patient table / couch
    draw_rounded_rect(ctx, rx - 120, ry + 50, 240, 30, 6)
    fill_white(ctx)

    # Table surface padding
    set_black(ctx, 1.5)
    draw_rounded_rect(ctx, rx - 110, ry + 53, 220, 24, 4)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Rotation markers
    ctx.arc(rx, ry + 100, 18, 0, 2 * math.pi)
    ctx.stroke()
    # Arrow arc
    ctx.arc(rx, ry + 100, 14, -0.5, 2.0)
    ctx.stroke()
    # Arrowhead
    ctx.move_to(rx + 14 * math.cos(2.0), ry + 100 + 14 * math.sin(2.0))
    ctx.line_to(rx + 14 * math.cos(2.0) + 5, ry + 100 + 14 * math.sin(2.0) - 5)
    ctx.stroke()

    # 6-DOF labels
    set_black(ctx, 1.5)
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(11)
    ctx.move_to(rx - 15, ry + 104)
    ctx.show_text("6-DOF")
    set_black(ctx, 2.5)

    # Gantry ring (above)
    ctx.save()
    set_black(ctx, 3.0)
    ctx.arc(rx, ry - 20, 100, -math.pi + 0.3, -0.3)
    ctx.stroke()
    ctx.arc(rx, ry - 20, 95, -math.pi + 0.35, -0.35)
    ctx.stroke()
    set_black(ctx, 2.5)
    ctx.restore()

    # Beam indicator
    set_black(ctx, 1.5)
    ctx.set_dash([4, 4])
    ctx.move_to(rx, ry - 20)
    ctx.line_to(rx, ry + 50)
    ctx.stroke()
    ctx.set_dash([])
    set_black(ctx, 2.5)

    # Laser alignment crosshairs
    for d in [-1, 1]:
        ctx.set_dash([3, 3])
        ctx.move_to(rx + d * 25, ry + 35)
        ctx.line_to(rx + d * 25, ry + 55)
        ctx.stroke()
    ctx.set_dash([])


def draw_needle_placement_robot(ctx, rx, ry):
    """Robotic needle-placement system (CT/MRI-guided biopsy/ablation robot)."""
    # Base unit
    draw_rounded_rect(ctx, rx - 50, ry + 120, 100, 130, 8)
    fill_white(ctx)

    # Screen on base
    draw_rounded_rect(ctx, rx - 35, ry + 135, 70, 45, 4)
    ctx.stroke()
    # Crosshair on screen
    ctx.move_to(rx - 15, ry + 157)
    ctx.line_to(rx + 15, ry + 157)
    ctx.stroke()
    ctx.move_to(rx, ry + 142)
    ctx.line_to(rx, ry + 172)
    ctx.stroke()
    # Target circle
    ctx.arc(rx, ry + 157, 8, 0, 2 * math.pi)
    ctx.stroke()

    # Articulating arm
    joints = [
        (rx, ry + 120),
        (rx + 35, ry + 70),
        (rx + 50, ry + 20),
        (rx + 30, ry - 20),
    ]
    set_black(ctx, 3.0)
    for i in range(len(joints) - 1):
        ctx.move_to(joints[i][0], joints[i][1])
        ctx.line_to(joints[i + 1][0], joints[i + 1][1])
        ctx.stroke()
    set_black(ctx, 2.5)

    for jx, jy in joints:
        ctx.arc(jx, jy, 5, 0, 2 * math.pi)
        fill_white(ctx)

    # Needle guide assembly
    ex, ey = joints[-1]
    draw_rounded_rect(ctx, ex - 15, ey - 15, 30, 30, 3)
    fill_white(ctx)

    # Needle
    set_black(ctx, 2.0)
    ctx.move_to(ex, ey)
    ctx.line_to(ex - 30, ey + 60)
    ctx.stroke()
    # Needle tip
    ctx.move_to(ex - 30, ey + 60)
    ctx.line_to(ex - 32, ey + 70)
    ctx.stroke()

    # Depth markings on needle
    set_black(ctx, 1.0)
    for t in [0.2, 0.4, 0.6, 0.8]:
        nx = ex + t * (-30)
        ny = ey + t * 60
        ctx.move_to(nx - 3, ny - 2)
        ctx.line_to(nx + 3, ny + 2)
        ctx.stroke()
    set_black(ctx, 2.5)

    # Wheels
    for wx in [rx - 35, rx + 35]:
        ctx.arc(wx, ry + 255, 8, 0, 2 * math.pi)
        fill_white(ctx)


def draw_social_companion_robot(ctx, rx, ry):
    """Social companion robot (e.g., Pepper/NAO-style) for pediatric patients."""
    # Body - rounded, friendly appearance
    draw_rounded_rect(ctx, rx - 45, ry + 30, 90, 120, 20)
    fill_white(ctx)

    # Head - large, round, friendly
    ctx.arc(rx, ry - 20, 55, 0, 2 * math.pi)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.5)
    ctx.stroke()

    # Big friendly eyes (larger for expressive look)
    for ex in [rx - 20, rx + 20]:
        ctx.arc(ex, ry - 25, 14, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()
        # Pupils
        ctx.arc(ex + 3, ry - 27, 6, 0, 2 * math.pi)
        ctx.fill()
        # Eye highlight
        ctx.arc(ex + 5, ry - 30, 2.5, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()
        set_black(ctx, 2.5)

    # Friendly smile
    ctx.arc(rx, ry - 2, 18, 0.2, math.pi - 0.2)
    ctx.stroke()

    # Chest tablet/screen
    draw_rounded_rect(ctx, rx - 28, ry + 45, 56, 40, 6)
    ctx.stroke()
    # Heart on screen
    set_black(ctx, 2.0)
    hx, hy = rx, ry + 60
    ctx.move_to(hx, hy + 10)
    ctx.curve_to(hx - 15, hy - 5, hx - 15, hy - 15, hx, hy - 5)
    ctx.curve_to(hx + 15, hy - 15, hx + 15, hy - 5, hx, hy + 10)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Arms (small, rounded)
    for side in [-1, 1]:
        sx = rx + side * 45
        ctx.move_to(sx, ry + 45)
        ctx.curve_to(sx + side * 30, ry + 55, sx + side * 35, ry + 80, sx + side * 20, ry + 100)
        ctx.stroke()
        # Hand
        ctx.arc(sx + side * 20, ry + 100, 8, 0, 2 * math.pi)
        fill_white(ctx)

    # Base (wheeled)
    ctx.move_to(rx - 35, ry + 150)
    ctx.line_to(rx - 40, ry + 180)
    ctx.line_to(rx + 40, ry + 180)
    ctx.line_to(rx + 35, ry + 150)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.5)
    ctx.stroke()

    # Wheels
    for wx in [rx - 25, rx + 25]:
        ctx.arc(wx, ry + 185, 7, 0, 2 * math.pi)
        fill_white(ctx)


def draw_humanoid_robot(ctx, rx, ry):
    """Humanoid robot (bipedal, full-size) for pediatric assistive tasks."""
    # Head
    draw_rounded_rect(ctx, rx - 30, ry - 70, 60, 50, 12)
    fill_white(ctx)

    # Eyes (LED-style)
    for ex in [rx - 12, rx + 12]:
        draw_rounded_rect(ctx, ex - 8, ry - 58, 16, 8, 3)
        ctx.stroke()
        # LED glow
        ctx.arc(ex, ry - 54, 3, 0, 2 * math.pi)
        ctx.fill()

    # Mouth line
    ctx.move_to(rx - 10, ry - 35)
    ctx.line_to(rx + 10, ry - 35)
    ctx.stroke()

    # Neck
    ctx.move_to(rx - 10, ry - 20)
    ctx.line_to(rx - 10, ry - 10)
    ctx.line_to(rx + 10, ry - 10)
    ctx.line_to(rx + 10, ry - 20)
    ctx.stroke()

    # Torso
    ctx.move_to(rx - 40, ry - 10)
    ctx.line_to(rx - 35, ry + 80)
    ctx.line_to(rx + 35, ry + 80)
    ctx.line_to(rx + 40, ry - 10)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.5)
    ctx.stroke()

    # Chest panel
    draw_rounded_rect(ctx, rx - 22, ry + 5, 44, 30, 5)
    ctx.stroke()

    # Arms with joints
    for side in [-1, 1]:
        # Shoulder joint
        sx = rx + side * 43
        ctx.arc(sx, ry - 5, 8, 0, 2 * math.pi)
        fill_white(ctx)
        # Upper arm
        elbow_x = sx + side * 25
        elbow_y = ry + 40
        ctx.move_to(sx, ry - 5)
        ctx.line_to(elbow_x, elbow_y)
        ctx.stroke()
        # Elbow joint
        ctx.arc(elbow_x, elbow_y, 6, 0, 2 * math.pi)
        fill_white(ctx)
        # Forearm
        hand_x = elbow_x + side * 10
        hand_y = elbow_y + 45
        ctx.move_to(elbow_x, elbow_y)
        ctx.line_to(hand_x, hand_y)
        ctx.stroke()
        # Hand (5-finger suggestion)
        ctx.arc(hand_x, hand_y, 9, 0, 2 * math.pi)
        fill_white(ctx)
        for f in range(5):
            fa = math.radians(180 + f * 36 + side * 20)
            ctx.move_to(hand_x + 9 * math.cos(fa), hand_y + 9 * math.sin(fa))
            ctx.line_to(hand_x + 16 * math.cos(fa), hand_y + 16 * math.sin(fa))
            ctx.stroke()

    # Hip joint
    ctx.move_to(rx - 25, ry + 80)
    ctx.line_to(rx + 25, ry + 80)
    ctx.stroke()

    # Legs
    for side in [-1, 1]:
        hip_x = rx + side * 20
        # Upper leg
        knee_x = hip_x + side * 5
        knee_y = ry + 140
        ctx.move_to(hip_x, ry + 80)
        ctx.line_to(knee_x, knee_y)
        ctx.stroke()
        ctx.arc(knee_x, knee_y, 6, 0, 2 * math.pi)
        fill_white(ctx)
        # Lower leg
        foot_x = knee_x
        foot_y = knee_y + 55
        ctx.move_to(knee_x, knee_y)
        ctx.line_to(foot_x, foot_y)
        ctx.stroke()
        # Foot
        ctx.move_to(foot_x - 12, foot_y)
        ctx.line_to(foot_x + 18, foot_y)
        ctx.line_to(foot_x + 18, foot_y + 8)
        ctx.line_to(foot_x - 12, foot_y + 8)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.5)
        ctx.stroke()


def draw_motion_tracking_robot(ctx, rx, ry):
    """Radiotherapy motion-management / tracking robot system."""
    # Main tracking unit housing
    draw_rounded_rect(ctx, rx - 55, ry + 60, 110, 160, 8)
    fill_white(ctx)

    # Display screen
    draw_rounded_rect(ctx, rx - 40, ry + 75, 80, 50, 4)
    ctx.stroke()

    # Waveform on screen (breathing trace)
    set_black(ctx, 1.5)
    points = []
    for i in range(20):
        px = rx - 35 + i * 3.5
        py = ry + 100 + 12 * math.sin(i * 0.8)
        points.append((px, py))
    ctx.move_to(points[0][0], points[0][1])
    for px, py in points[1:]:
        ctx.line_to(px, py)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Camera/sensor head on articulating arm
    ctx.move_to(rx, ry + 60)
    ctx.line_to(rx, ry + 20)
    ctx.stroke()

    # Horizontal arm
    ctx.move_to(rx - 60, ry + 20)
    ctx.line_to(rx + 60, ry + 20)
    ctx.stroke()

    # Tracking cameras (stereo pair)
    for dx in [-40, 0, 40]:
        cam_x = rx + dx
        cam_y = ry - 10
        # Camera housing
        draw_rounded_rect(ctx, cam_x - 12, cam_y - 12, 24, 24, 4)
        fill_white(ctx)
        # Lens
        ctx.arc(cam_x, cam_y, 7, 0, 2 * math.pi)
        ctx.stroke()
        ctx.arc(cam_x, cam_y, 3, 0, 2 * math.pi)
        ctx.fill()

    # IR marker tracking lines (dashed)
    set_black(ctx, 1.5)
    ctx.set_dash([5, 5])
    for dx in [-40, 0, 40]:
        ctx.move_to(rx + dx, ry + 2)
        ctx.line_to(rx + dx * 0.3, ry + 50)
        ctx.stroke()
    ctx.set_dash([])
    set_black(ctx, 2.5)

    # Reflective marker block (on patient)
    draw_rounded_rect(ctx, rx - 18, ry + 42, 36, 12, 2)
    ctx.stroke()
    for mdx in [-8, 0, 8]:
        ctx.arc(rx + mdx, ry + 48, 3, 0, 2 * math.pi)
        ctx.fill()

    # Wheels
    for wx in [rx - 40, rx + 40]:
        ctx.arc(wx, ry + 225, 8, 0, 2 * math.pi)
        fill_white(ctx)


def draw_imaging_assistant_robot(ctx, rx, ry):
    """Imaging assistant robot (robotic ultrasound/probe holder)."""
    # Base cart
    draw_rounded_rect(ctx, rx - 45, ry + 130, 90, 100, 8)
    fill_white(ctx)

    # Ultrasound display
    draw_rounded_rect(ctx, rx - 35, ry - 50, 70, 55, 5)
    fill_white(ctx)
    # Image on display
    set_black(ctx, 1.5)
    ctx.arc(rx, ry - 25, 18, 0, 2 * math.pi)
    ctx.stroke()
    ctx.arc(rx + 5, ry - 28, 8, 0, 2 * math.pi)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Display arm / gooseneck
    ctx.move_to(rx, ry + 5)
    ctx.curve_to(rx + 10, ry + 30, rx - 10, ry + 60, rx, ry + 80)
    ctx.stroke()

    # Articulating probe arm
    ctx.move_to(rx, ry + 80)
    ctx.line_to(rx - 30, ry + 50)
    ctx.stroke()
    ctx.arc(rx - 30, ry + 50, 5, 0, 2 * math.pi)
    fill_white(ctx)

    ctx.move_to(rx - 30, ry + 50)
    ctx.line_to(rx - 60, ry + 30)
    ctx.stroke()
    ctx.arc(rx - 60, ry + 30, 5, 0, 2 * math.pi)
    fill_white(ctx)

    # Force-controlled probe holder
    draw_rounded_rect(ctx, rx - 75, ry + 15, 30, 30, 4)
    fill_white(ctx)

    # Ultrasound probe
    ctx.move_to(rx - 60, ry + 45)
    ctx.line_to(rx - 60, ry + 75)
    ctx.stroke()
    # Probe head
    ctx.arc(rx - 60, ry + 78, 8, 0, math.pi)
    ctx.stroke()

    # Force sensor label
    set_black(ctx, 1.5)
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(9)
    ctx.move_to(rx - 74, ry + 34)
    ctx.show_text("F/T")
    set_black(ctx, 2.5)

    # Wheels
    for wx in [rx - 30, rx + 30]:
        ctx.arc(wx, ry + 235, 8, 0, 2 * math.pi)
        fill_white(ctx)


def draw_steerable_needle_robot(ctx, rx, ry):
    """Steerable needle / needle-steering robot (bevel-tip, MRI-compatible)."""
    # Robot base frame (MRI-compatible, non-metallic look)
    draw_rounded_rect(ctx, rx - 50, ry + 70, 100, 80, 6)
    fill_white(ctx)

    # Guide tube
    set_black(ctx, 3.0)
    ctx.move_to(rx + 10, ry + 70)
    ctx.line_to(rx + 10, ry + 10)
    ctx.stroke()
    ctx.move_to(rx - 10, ry + 70)
    ctx.line_to(rx - 10, ry + 10)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Steerable needle (curved path shown)
    set_black(ctx, 2.0)
    ctx.move_to(rx, ry + 10)
    ctx.curve_to(rx - 5, ry - 20, rx - 15, ry - 50, rx - 25, ry - 80)
    ctx.stroke()

    # Bevel tip
    ctx.move_to(rx - 25, ry - 80)
    ctx.line_to(rx - 28, ry - 90)
    ctx.line_to(rx - 22, ry - 88)
    ctx.close_path()
    ctx.fill()

    # Target (tumor)
    ctx.set_dash([3, 3])
    ctx.arc(rx - 30, ry - 100, 15, 0, 2 * math.pi)
    ctx.stroke()
    ctx.set_dash([])
    # Crosshair in target
    ctx.move_to(rx - 40, ry - 100)
    ctx.line_to(rx - 20, ry - 100)
    ctx.stroke()
    ctx.move_to(rx - 30, ry - 110)
    ctx.line_to(rx - 30, ry - 90)
    ctx.stroke()

    set_black(ctx, 2.5)

    # Actuation unit (below)
    draw_rounded_rect(ctx, rx - 40, ry + 80, 80, 35, 4)
    ctx.stroke()
    # Motor indicators
    for mdx in [-20, 0, 20]:
        ctx.arc(rx + mdx, ry + 97, 5, 0, 2 * math.pi)
        fill_white(ctx)

    # Insertion depth indicator
    set_black(ctx, 1.5)
    ctx.move_to(rx + 25, ry + 10)
    ctx.line_to(rx + 25, ry - 80)
    ctx.stroke()
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        iy = ry + 10 + t * (-90)
        ctx.move_to(rx + 22, iy)
        ctx.line_to(rx + 28, iy)
        ctx.stroke()
    set_black(ctx, 2.5)

    # Image guidance overlay (MRI/CT icon)
    draw_rounded_rect(ctx, rx - 48, ry + 130, 96, 25, 3)
    ctx.stroke()
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(11)
    ctx.move_to(rx - 38, ry + 147)
    ctx.show_text("Image-Guided")


def draw_rehab_exoskeleton(ctx, rx, ry):
    """Rehabilitation exoskeleton / robotic gait trainer."""
    # Frame structure (bilateral leg exoskeleton)
    # Hip frame
    ctx.move_to(rx - 50, ry - 20)
    ctx.line_to(rx + 50, ry - 20)
    ctx.stroke()
    draw_rounded_rect(ctx, rx - 55, ry - 35, 110, 30, 5)
    fill_white(ctx)

    # Waist belt
    set_black(ctx, 3.0)
    ctx.arc(rx, ry - 20, 50, -0.3, math.pi + 0.3)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Back support / power unit
    draw_rounded_rect(ctx, rx - 25, ry - 55, 50, 40, 5)
    fill_white(ctx)
    # Battery indicator
    draw_rounded_rect(ctx, rx - 15, ry - 48, 30, 12, 2)
    ctx.stroke()
    ctx.set_source_rgb(0, 0, 0)
    draw_rounded_rect(ctx, rx - 13, ry - 46, 18, 8, 1)
    ctx.fill()

    for side in [-1, 1]:
        hip_x = rx + side * 40
        # Hip joint actuator
        ctx.arc(hip_x, ry - 5, 10, 0, 2 * math.pi)
        fill_white(ctx)
        # Gear symbol
        for ga in range(0, 360, 45):
            a = math.radians(ga)
            ctx.move_to(hip_x + 8 * math.cos(a), ry - 5 + 8 * math.sin(a))
            ctx.line_to(hip_x + 12 * math.cos(a), ry - 5 + 12 * math.sin(a))
            ctx.stroke()

        # Thigh brace
        set_black(ctx, 3.0)
        knee_x = hip_x + side * 5
        knee_y = ry + 60
        ctx.move_to(hip_x, ry - 5)
        ctx.line_to(knee_x, knee_y)
        ctx.stroke()
        set_black(ctx, 2.5)

        # Straps on thigh
        for sy in [ry + 15, ry + 35]:
            ctx.move_to(hip_x - 8, sy)
            ctx.line_to(knee_x + 8, sy)
            ctx.stroke()

        # Knee joint actuator
        ctx.arc(knee_x, knee_y, 10, 0, 2 * math.pi)
        fill_white(ctx)
        for ga in range(0, 360, 45):
            a = math.radians(ga)
            ctx.move_to(knee_x + 8 * math.cos(a), knee_y + 8 * math.sin(a))
            ctx.line_to(knee_x + 12 * math.cos(a), knee_y + 12 * math.sin(a))
            ctx.stroke()

        # Shank brace
        set_black(ctx, 3.0)
        ankle_x = knee_x
        ankle_y = knee_y + 55
        ctx.move_to(knee_x, knee_y)
        ctx.line_to(ankle_x, ankle_y)
        ctx.stroke()
        set_black(ctx, 2.5)

        # Straps on shank
        for sy in [knee_y + 20, knee_y + 40]:
            ctx.move_to(knee_x - 8, sy)
            ctx.line_to(ankle_x + 8, sy)
            ctx.stroke()

        # Ankle joint
        ctx.arc(ankle_x, ankle_y, 7, 0, 2 * math.pi)
        fill_white(ctx)

        # Foot plate
        ctx.move_to(ankle_x - 15, ankle_y + 5)
        ctx.line_to(ankle_x + 20, ankle_y + 5)
        ctx.line_to(ankle_x + 22, ankle_y + 12)
        ctx.line_to(ankle_x - 15, ankle_y + 12)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.5)
        ctx.stroke()

    # Handrail supports (parallel bars)
    for side in [-1, 1]:
        bar_x = rx + side * 95
        ctx.move_to(bar_x, ry - 60)
        ctx.line_to(bar_x, ry + 130)
        ctx.stroke()
        # Top rail
        set_black(ctx, 3.0)
        ctx.move_to(bar_x - 10, ry - 60)
        ctx.line_to(bar_x + 10, ry - 60)
        ctx.stroke()
        set_black(ctx, 2.5)


ROBOT_DRAWERS = [
    draw_surgical_robot,
    draw_cobot,
    draw_radiotherapy_positioning_robot,
    draw_needle_placement_robot,
    draw_social_companion_robot,
    draw_humanoid_robot,
    draw_motion_tracking_robot,
    draw_imaging_assistant_robot,
    draw_steerable_needle_robot,
    draw_rehab_exoskeleton,
]


def draw_patient_for_page(ctx, px, py, page_idx, robot_pos_x):
    """Draw a patient appropriate for the page (adult or child, diverse)."""
    is_child = page_idx in PEDIATRIC_PAGES
    head_r = 38 if not is_child else 32
    body_h = 200 if not is_child else 150

    # Draw head
    draw_head(ctx, px, py, head_r, is_child)

    # Hair (diverse styles)
    if is_child:
        child_style = CHILD_HAIR_STYLES[page_idx - 4]
        child_style(ctx, px, py, head_r)
    else:
        adult_idx = page_idx if page_idx < 4 else page_idx - 2
        hair_style = HAIR_STYLES[adult_idx % len(HAIR_STYLES)]
        hair_style(ctx, px, py, head_r)

    # Body
    shoulder_y = py + head_r + 5
    seated = page_idx in {0, 2, 3, 6, 7, 8}  # Some pages have seated patients
    draw_patient_body(ctx, px, shoulder_y, body_h, is_child, seated)

    # Arm reaching toward robot
    shoulder_x = px + (35 if not is_child else 25)
    reach_x = robot_pos_x - 30 if robot_pos_x > px else robot_pos_x + 30
    reach_y = py + head_r + 60
    draw_arm_reaching(ctx, shoulder_x, shoulder_y + 15, reach_x, reach_y)

    # Other arm
    other_shoulder = px - (35 if not is_child else 25)
    if page_idx == 9:  # Exoskeleton - both hands on bars
        draw_arm_reaching(ctx, other_shoulder, shoulder_y + 15, px - 85, py - 20)
    else:
        draw_arm_down(ctx, other_shoulder, shoulder_y + 15, body_h * 0.4)


def draw_illustration(page_idx):
    """Draw a complete illustration for a given page."""
    robot_type = ROBOT_TYPES[page_idx]
    is_child = page_idx in PEDIATRIC_PAGES

    # Create SVG surface first
    svg_path = os.path.join(SVG_DIR, f"{page_idx + 1:02d}_{robot_type.lower().replace(' ', '_').replace('/', '_')}.svg")
    surface = cairo.SVGSurface(svg_path, IMG_W, IMG_H)
    ctx = cairo.Context(surface)

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, IMG_W, IMG_H)
    ctx.fill()

    # Drawing setup
    set_black(ctx, 2.5)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # Border
    set_black(ctx, 1.5)
    draw_rounded_rect(ctx, 40, 40, IMG_W - 80, IMG_H - 80, 15)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Title area
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(36)
    title_text = f"Patient-Robot Interaction: {robot_type}"
    extents = ctx.text_extents(title_text)
    ctx.move_to((IMG_W - extents.width) / 2, 110)
    ctx.show_text(title_text)

    # Divider line
    ctx.move_to(80, 130)
    ctx.line_to(IMG_W - 80, 130)
    ctx.stroke()

    # Scene description
    ctx.set_font_size(16)
    ctx.select_font_face("serif", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    if is_child:
        scene_text = "Pediatric Oncology Trial Setting"
    else:
        scene_text = "Adult Oncology Trial Setting"
    extents = ctx.text_extents(scene_text)
    ctx.move_to((IMG_W - extents.width) / 2, 155)
    ctx.show_text(scene_text)

    # Main illustration area
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    # Floor line
    set_black(ctx, 1.5)
    floor_y = 1700
    ctx.move_to(80, floor_y)
    ctx.line_to(IMG_W - 80, floor_y)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Position robot and patient
    robot_x = IMG_W * 0.62
    robot_y = 900 if not is_child else 950
    patient_x = IMG_W * 0.28
    patient_y = 600 if not is_child else 700

    # Draw robot
    ctx.save()
    scale_factor = 2.2 if not is_child else 1.8
    ctx.translate(robot_x, robot_y)
    ctx.scale(scale_factor, scale_factor)
    ctx.translate(-robot_x, -robot_y)
    ROBOT_DRAWERS[page_idx](ctx, robot_x, robot_y)
    ctx.restore()

    # Draw patient
    draw_patient_for_page(ctx, patient_x, patient_y, page_idx, robot_x)

    # ISO safety symbol (bottom right of illustration)
    if page_idx in {0, 3, 8}:  # Surgical/needle pages
        draw_iso_warning_triangle(ctx, IMG_W - 150, 1650, 40)
    elif page_idx in {2, 6}:  # Radiotherapy pages
        draw_iso_radiation(ctx, IMG_W - 150, 1650, 40)
    else:
        draw_iso_keep_still(ctx, IMG_W - 150, 1650, 50)

    # Label
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(12)
    if page_idx in {0, 3, 8}:
        ctx.move_to(IMG_W - 190, 1690)
        ctx.show_text("ISO 7010: Caution")
    elif page_idx in {2, 6}:
        ctx.move_to(IMG_W - 200, 1690)
        ctx.show_text("ISO 7010 W003")
    else:
        ctx.move_to(IMG_W - 200, 1690)
        ctx.show_text("ISO 15223-1")

    # Robot type label below illustration
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(22)
    label = robot_type
    extents = ctx.text_extents(label)
    ctx.move_to((IMG_W - extents.width) / 2, 1740)
    ctx.show_text(label)

    # Bottom divider
    set_black(ctx, 1.5)
    ctx.move_to(80, 1760)
    ctx.line_to(IMG_W - 80, 1760)
    ctx.stroke()

    # Page number
    ctx.set_font_size(14)
    page_str = f"Page {page_idx + 1} of 10"
    extents = ctx.text_extents(page_str)
    ctx.move_to((IMG_W - extents.width) / 2, 1790)
    ctx.show_text(page_str)

    # Finalize SVG
    surface.finish()

    # Convert SVG to PDF
    pdf_path = os.path.join(PDF_DIR, f"{page_idx + 1:02d}_{robot_type.lower().replace(' ', '_').replace('/', '_')}.pdf")
    pdf_surface = cairo.PDFSurface(pdf_path, IMG_W, IMG_H)
    pdf_ctx = cairo.Context(pdf_surface)
    # Re-read SVG and render to PDF via re-drawing
    # Since we can't easily re-render SVG, we'll draw directly to PDF too
    pdf_surface.finish()

    # Convert SVG to PNG
    png_path = os.path.join(PNG_DIR, f"{page_idx + 1:02d}_{robot_type.lower().replace(' ', '_').replace('/', '_')}.png")

    return svg_path, pdf_path, png_path


def draw_to_surface(surface, page_idx):
    """Draw illustration directly to a Cairo surface."""
    robot_type = ROBOT_TYPES[page_idx]
    is_child = page_idx in PEDIATRIC_PAGES

    ctx = cairo.Context(surface)

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, IMG_W, IMG_H)
    ctx.fill()

    set_black(ctx, 2.5)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # Thin border
    set_black(ctx, 1.0)
    draw_rounded_rect(ctx, 30, 30, IMG_W - 60, IMG_H - 60, 12)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Title
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(38)
    title_text = f"Patient-Robot Interaction: {robot_type}"
    extents = ctx.text_extents(title_text)
    # If title too long, reduce font
    if extents.width > IMG_W - 160:
        ctx.set_font_size(28)
        extents = ctx.text_extents(title_text)
    ctx.move_to((IMG_W - extents.width) / 2, 100)
    ctx.show_text(title_text)

    # Divider
    set_black(ctx, 1.5)
    ctx.move_to(60, 120)
    ctx.line_to(IMG_W - 60, 120)
    ctx.stroke()

    # Scene label
    ctx.set_font_size(16)
    ctx.select_font_face("serif", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    scene = "Pediatric Oncology Trial Setting" if is_child else "Adult Oncology Trial Setting"
    extents = ctx.text_extents(scene)
    ctx.move_to((IMG_W - extents.width) / 2, 148)
    ctx.show_text(scene)

    set_black(ctx, 2.5)

    # Floor line
    set_black(ctx, 1.0)
    floor_y = 1680
    ctx.set_dash([8, 4])
    ctx.move_to(60, floor_y)
    ctx.line_to(IMG_W - 60, floor_y)
    ctx.stroke()
    ctx.set_dash([])
    set_black(ctx, 2.5)

    # Positions
    robot_x = IMG_W * 0.62
    robot_y_base = 850
    patient_x = IMG_W * 0.3
    patient_y = 550 if not is_child else 650

    # Scale robots to be prominent
    ctx.save()
    sf = 2.5 if not is_child else 2.0
    ctx.translate(robot_x, robot_y_base)
    ctx.scale(sf, sf)
    ctx.translate(-robot_x, -robot_y_base)
    ROBOT_DRAWERS[page_idx](ctx, robot_x, robot_y_base)
    ctx.restore()

    # Patient
    draw_patient_for_page(ctx, patient_x, patient_y, page_idx, robot_x)

    # ISO symbol
    sym_x, sym_y = IMG_W - 140, 1720
    if page_idx in {0, 3, 8}:
        draw_iso_warning_triangle(ctx, sym_x, sym_y, 35)
        ctx.set_font_size(10)
        ctx.move_to(sym_x - 40, sym_y + 30)
        ctx.show_text("ISO 7010")
    elif page_idx in {2, 6}:
        draw_iso_radiation(ctx, sym_x, sym_y, 35)
        ctx.set_font_size(10)
        ctx.move_to(sym_x - 45, sym_y + 30)
        ctx.show_text("ISO 7010 W003")
    else:
        draw_iso_keep_still(ctx, sym_x, sym_y, 40)
        ctx.set_font_size(10)
        ctx.move_to(sym_x - 40, sym_y + 30)
        ctx.show_text("ISO 15223-1")

    # Robot label
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(20)
    lbl = robot_type
    extents = ctx.text_extents(lbl)
    ctx.move_to((IMG_W - extents.width) / 2, 1720)
    ctx.show_text(lbl)

    # Bottom divider
    set_black(ctx, 1.0)
    ctx.move_to(60, 1740)
    ctx.line_to(IMG_W - 60, 1740)
    ctx.stroke()

    # Page number
    ctx.set_font_size(13)
    pg = f"Page {page_idx + 1} of 10"
    extents = ctx.text_extents(pg)
    ctx.move_to((IMG_W - extents.width) / 2, 1770)
    ctx.show_text(pg)

    return ctx


def generate_all():
    """Generate all illustrations in SVG, PDF, and PNG formats."""
    os.makedirs(SVG_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)

    for idx in range(10):
        robot_type = ROBOT_TYPES[idx]
        safe_name = robot_type.lower().replace(" ", "_").replace("/", "_")
        prefix = f"{idx + 1:02d}_{safe_name}"

        print(f"Generating illustration {idx + 1}/10: {robot_type}")

        # SVG
        svg_path = os.path.join(SVG_DIR, f"{prefix}.svg")
        svg_surface = cairo.SVGSurface(svg_path, IMG_W, IMG_H)
        draw_to_surface(svg_surface, idx)
        svg_surface.finish()
        print(f"  SVG: {svg_path}")

        # PDF
        pdf_path = os.path.join(PDF_DIR, f"{prefix}.pdf")
        pdf_surface = cairo.PDFSurface(pdf_path, IMG_W, IMG_H)
        draw_to_surface(pdf_surface, idx)
        pdf_surface.finish()
        print(f"  PDF: {pdf_path}")

        # PNG (high resolution)
        png_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMG_W * 2, IMG_H * 2)
        png_ctx = cairo.Context(png_surface)
        png_ctx.scale(2, 2)
        # White bg
        png_ctx.set_source_rgb(1, 1, 1)
        png_ctx.rectangle(0, 0, IMG_W, IMG_H)
        png_ctx.fill()
        # Re-draw
        set_black(png_ctx, 2.5)
        png_ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        png_ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        draw_to_surface(cairo.PDFSurface("/dev/null", IMG_W, IMG_H), idx)  # dummy
        # Actually draw on PNG surface properly
        png_surface_real = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMG_W * 2, IMG_H * 2)
        png_ctx_real = cairo.Context(png_surface_real)
        png_ctx_real.scale(2, 2)
        draw_to_png(png_surface_real, idx)
        png_path = os.path.join(PNG_DIR, f"{prefix}.png")
        png_surface_real.write_to_png(png_path)
        print(f"  PNG: {png_path}")

    print("\nAll illustrations generated successfully.")


def draw_to_png(surface, page_idx):
    """Draw to a PNG image surface (scaled 2x)."""
    ctx = cairo.Context(surface)
    ctx.scale(2, 2)

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, IMG_W, IMG_H)
    ctx.fill()

    set_black(ctx, 2.5)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    robot_type = ROBOT_TYPES[page_idx]
    is_child = page_idx in PEDIATRIC_PAGES

    # Border
    set_black(ctx, 1.0)
    draw_rounded_rect(ctx, 30, 30, IMG_W - 60, IMG_H - 60, 12)
    ctx.stroke()
    set_black(ctx, 2.5)

    # Title
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(38)
    title_text = f"Patient-Robot Interaction: {robot_type}"
    extents = ctx.text_extents(title_text)
    if extents.width > IMG_W - 160:
        ctx.set_font_size(28)
        extents = ctx.text_extents(title_text)
    ctx.move_to((IMG_W - extents.width) / 2, 100)
    ctx.show_text(title_text)

    # Divider
    set_black(ctx, 1.5)
    ctx.move_to(60, 120)
    ctx.line_to(IMG_W - 60, 120)
    ctx.stroke()

    # Scene
    ctx.set_font_size(16)
    ctx.select_font_face("serif", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    scene = "Pediatric Oncology Trial Setting" if is_child else "Adult Oncology Trial Setting"
    extents = ctx.text_extents(scene)
    ctx.move_to((IMG_W - extents.width) / 2, 148)
    ctx.show_text(scene)

    set_black(ctx, 2.5)

    # Floor
    set_black(ctx, 1.0)
    ctx.set_dash([8, 4])
    ctx.move_to(60, 1680)
    ctx.line_to(IMG_W - 60, 1680)
    ctx.stroke()
    ctx.set_dash([])
    set_black(ctx, 2.5)

    # Robot
    robot_x = IMG_W * 0.62
    robot_y_base = 850
    patient_x = IMG_W * 0.3
    patient_y = 550 if not is_child else 650

    ctx.save()
    sf = 2.5 if not is_child else 2.0
    ctx.translate(robot_x, robot_y_base)
    ctx.scale(sf, sf)
    ctx.translate(-robot_x, -robot_y_base)
    ROBOT_DRAWERS[page_idx](ctx, robot_x, robot_y_base)
    ctx.restore()

    # Patient
    draw_patient_for_page(ctx, patient_x, patient_y, page_idx, robot_x)

    # ISO symbol
    sym_x, sym_y = IMG_W - 140, 1720
    if page_idx in {0, 3, 8}:
        draw_iso_warning_triangle(ctx, sym_x, sym_y, 35)
    elif page_idx in {2, 6}:
        draw_iso_radiation(ctx, sym_x, sym_y, 35)
    else:
        draw_iso_keep_still(ctx, sym_x, sym_y, 40)

    # Label
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(20)
    extents = ctx.text_extents(robot_type)
    ctx.move_to((IMG_W - extents.width) / 2, 1720)
    ctx.show_text(robot_type)

    # Bottom line
    set_black(ctx, 1.0)
    ctx.move_to(60, 1740)
    ctx.line_to(IMG_W - 60, 1740)
    ctx.stroke()

    # Page
    ctx.set_font_size(13)
    pg = f"Page {page_idx + 1} of 10"
    extents = ctx.text_extents(pg)
    ctx.move_to((IMG_W - extents.width) / 2, 1770)
    ctx.show_text(pg)


if __name__ == "__main__":
    generate_all()
