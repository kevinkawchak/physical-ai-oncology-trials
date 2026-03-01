#!/usr/bin/env python3
"""Generate the complete 10-page Patient-Robot Instructions PDF.

Uses Cairo to create a high-resolution portrait A4 PDF with:
- Header bar with author info
- Title with robot type
- Prominent black-and-white illustration
- Detailed patient instructions
- Footer bar with date, DOI, model, page number, and sources

Robot Types (ordered by commonality):
1. Surgical Robots
2. Cobots
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A4 dimensions in points (1 pt = 1/72 inch)
PAGE_W = 595.28  # 210mm
PAGE_H = 841.89  # 297mm

MARGIN_LEFT = 42
MARGIN_RIGHT = 42
MARGIN_TOP = 28
MARGIN_BOTTOM = 28
CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT

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

PEDIATRIC_PAGES = {4, 5}

# Page-specific source references (truncated links)
PAGE_SOURCES = [
    "intuitivesurgical.com/davinci; doi.org/10.1007/s11701-020-01183-x",
    "franka.de/research; doi.org/10.1109/LRA.2023.3270038",
    "accuray.com/cyberknife; doi.org/10.1016/j.ijrobp.2022.09.077",
    "doi.org/10.1109/TBME.2021.3138352; doi.org/10.1002/rcs.2584",
    "softbankrobotics.com/pepper; doi.org/10.1007/s12369-022-00926-y",
    "bostondynamics.com/atlas; doi.org/10.1109/LRA.2024.3352367",
    "varian.com/respiratory-gating; doi.org/10.1016/j.radonc.2023.109924",
    "doi.org/10.1109/TMRB.2022.3213735; doi.org/10.1002/mp.15776",
    "doi.org/10.1177/02783649231225258; doi.org/10.1109/TRO.2023.3312697",
    "doi.org/10.1186/s12984-023-01275-3; eksobionics.com/ekso-indego",
]


# ── Instruction data for each robot type ─────────────────────
INSTRUCTIONS = [
    {  # Page 1: Surgical Robots
        "prep": [
            "Fast for 8 hours before your scheduled procedure (no food or drink).",
            "Shower with antibacterial soap the evening before and morning of.",
            "Wear loose, comfortable clothing; leave jewelry and valuables at home.",
            "Bring your photo ID, insurance card, and signed consent forms.",
            "Take prescribed pre-operative medications as directed by your oncologist.",
        ],
        "enter": [
            "A staff member will guide you to the pre-op area; the robot will already be in the OR.",
            "Change into the hospital gown provided and place belongings in the labeled bag.",
            "Lie on the operating table face-up when instructed.",
            "Place both arms on the padded arm rests, palms facing up.",
            "Remain still while the anesthesiologist administers sedation (approx. 2 min).",
        ],
        "interact": [
            "The surgical robot has 3--4 arms that will operate through small incisions (8--12 mm).",
            "You will be under general anesthesia and will not feel or need to do anything.",
            "Estimated procedure time: 45--180 minutes depending on tumor location.",
            "The surgeon controls the robot from a console nearby; the robot does not act autonomously.",
            "Robotic instruments include a camera, graspers, and cautery tools.",
        ],
        "conclude": [
            "After the procedure, the robot arms are retracted and incisions are closed.",
            "You will be moved to the recovery room; expect to wake in 15--30 minutes.",
            "A nurse will check your vitals every 15 minutes for the first hour.",
            "You may feel mild soreness at incision sites; request pain management as needed.",
        ],
        "followup": [
            "Keep incision sites clean and dry for 48 hours; change dressings as instructed.",
            "Attend your post-operative follow-up appointment in 7--10 days.",
            "Report any fever above 38.5 C, excessive bleeding, or unusual swelling immediately.",
            "Resume light activity after 1--2 weeks; avoid heavy lifting for 4--6 weeks.",
            "Your next robotic-assisted session (if needed) will be scheduled at follow-up.",
        ],
    },
    {  # Page 2: Cobots
        "prep": [
            "Eat a light meal 2 hours before your appointment to maintain comfort.",
            "Wear a short-sleeved shirt or top that allows arm and shoulder access.",
            "Review your medication list; bring any prescribed topical numbing cream.",
            "Arrive 15 minutes early to complete check-in paperwork.",
            "Remove watches, bracelets, and rings from the arm the cobot will assist.",
        ],
        "enter": [
            "Enter the procedure room; the cobot is mounted on a mobile cart beside the exam chair.",
            "Sit in the adjustable exam chair and place your arm on the padded arm rest.",
            "The cobot will be in standby mode with its arm retracted and a green status light on.",
            "A technician will verbally confirm the procedure and activate the cobot.",
            "Place your hand palm-down on the marked target area on the arm rest.",
        ],
        "interact": [
            "The cobot arm moves slowly and gently (max speed 250 mm/s); you may feel light pressure.",
            "If you feel discomfort, say 'Stop' clearly; the cobot will halt within 0.5 seconds.",
            "Estimated interaction time: 10--25 minutes for sample collection or positioning tasks.",
            "Keep your arm relaxed and still; the cobot uses force sensors to avoid excess pressure.",
            "The cobot may reposition your arm 2--3 times; follow verbal audio prompts.",
        ],
        "conclude": [
            "The cobot arm will retract to its home position when the task is complete.",
            "Remain seated for 2 minutes while the technician reviews results on screen.",
            "A bandage will be applied to any sample collection site.",
            "You will receive a printed summary of the session before leaving.",
        ],
        "followup": [
            "No special home care is usually required after cobot-assisted procedures.",
            "If a sample was collected, results will be available in 3--5 business days.",
            "Monitor the site for 24 hours; apply gentle pressure if any minor bleeding.",
            "Your next cobot session will be scheduled during your regular oncology visit.",
        ],
    },
    {  # Page 3: Radiotherapy Patient-Positioning Robots
        "prep": [
            "Wear comfortable clothing without metal (zippers, snaps, underwire).",
            "Do not apply lotion, deodorant, or powder to the treatment area.",
            "Eat and drink normally unless otherwise directed by your radiation oncologist.",
            "Take anxiety-reducing medication 30 min before if prescribed.",
            "Use the restroom before your session; a full bladder may be required for some sites.",
        ],
        "enter": [
            "Enter the treatment room; the robotic couch is the flat table in the center.",
            "Lie on the robotic couch in the position demonstrated during your simulation session.",
            "Place your arms in the arm cradle or above your head as instructed.",
            "A custom immobilization mask or mold may be placed to hold you steady.",
            "The couch will adjust automatically using 6 degrees of freedom (6-DOF).",
        ],
        "interact": [
            "The robotic couch shifts in small increments (1--3 mm) to align you with the beam.",
            "You will hear mechanical sounds as the couch moves; this is normal.",
            "Breathe normally and remain completely still during alignment (approx. 5--8 min).",
            "Laser crosshairs will project onto your skin to verify position; do not block them.",
            "Estimated total session time: 15--30 minutes including setup and treatment.",
        ],
        "conclude": [
            "After the beam delivery ends, the couch will lower to exit height.",
            "Wait for the therapist to remove any immobilization devices before sitting up.",
            "Sit up slowly to avoid dizziness and swing your legs off the side.",
            "The therapist will confirm your next session date and time.",
        ],
        "followup": [
            "Gently wash the treatment area with mild soap; do not scrub.",
            "Apply recommended moisturizer to the treatment area if skin redness develops.",
            "Attend all scheduled daily sessions (typically Mon--Fri for 3--7 weeks).",
            "Report skin changes, pain, or fatigue to your radiation oncology team.",
        ],
    },
    {  # Page 4: Robotic Needle-Placement Systems
        "prep": [
            "Fast for 4 hours before the procedure (clear liquids may be allowed).",
            "Inform the team of any blood-thinning medications; some may need to be paused.",
            "Wear loose clothing that provides access to the biopsy or treatment area.",
            "Arrange for someone to drive you home if sedation will be used.",
            "Bring prior imaging (CT/MRI scans) on CD or USB if requested.",
        ],
        "enter": [
            "Enter the procedure room; the robotic needle system is beside the imaging table.",
            "Lie on the imaging table in the position indicated (face-up or face-down).",
            "Place both hands at your sides or above your head as directed.",
            "A local anesthetic will be applied to the needle insertion site (1--2 min to numb).",
            "The robot's articulating arm will position the needle guide over the target.",
        ],
        "interact": [
            "The robot aligns the needle to sub-millimeter accuracy using real-time imaging.",
            "You may feel pressure but should not feel sharp pain; alert staff if you do.",
            "Hold your breath briefly (5--10 seconds) when instructed for image capture.",
            "The needle insertion takes 1--3 minutes; the robot maintains steady guidance.",
            "Estimated total procedure time: 20--45 minutes including imaging and sampling.",
        ],
        "conclude": [
            "The needle is withdrawn and gentle pressure is applied for 3--5 minutes.",
            "A sterile bandage is placed over the insertion site.",
            "Remain lying down for 15--30 minutes for post-procedure monitoring.",
            "A nurse will check your blood pressure and verify no immediate complications.",
        ],
        "followup": [
            "Keep the bandage dry for 24 hours; replace with a clean one after.",
            "Avoid strenuous activity or heavy lifting for 48 hours.",
            "Biopsy results are typically available within 5--7 business days.",
            "Contact your care team if you experience increasing pain, swelling, or bleeding.",
            "A follow-up appointment will be scheduled to review results and next steps.",
        ],
    },
    {  # Page 5: Social Companion Robots (PEDIATRIC)
        "prep": [
            "Parents/guardians: explain to your child that they will meet a friendly robot.",
            "Bring your child's favorite small comfort item (stuffed animal, blanket).",
            "Ensure your child has eaten a light snack and is well-hydrated.",
            "Dress your child in comfortable, easy-to-move-in clothing.",
            "Review the picture book provided at your last visit showing the robot.",
        ],
        "enter": [
            "Your child enters the room alone (you may watch through the window).",
            "The companion robot (about 1.2 m tall) will greet your child by name.",
            "The robot's screen displays a friendly face and a welcome message.",
            "Encourage your child to wave or say 'Hello' to the robot.",
            "The robot will extend its hand for a gentle high-five if your child is comfortable.",
        ],
        "interact": [
            "The robot guides your child through breathing exercises with visual cues on its screen.",
            "Your child can talk to the robot; it responds with simple, comforting phrases.",
            "Interactive games on the robot's chest screen help distract during waiting (5--10 min).",
            "The robot demonstrates how to sit still using fun animations (e.g., 'freeze like a statue').",
            "Estimated interaction time: 10--20 minutes before and/or during the clinical procedure.",
        ],
        "conclude": [
            "The robot will say goodbye and offer a virtual sticker on its screen.",
            "Your child can give the robot a high-five or wave goodbye.",
            "The robot plays a short congratulatory song (15 seconds).",
            "A staff member will escort your child back to you.",
        ],
        "followup": [
            "Ask your child about their experience to reinforce positive associations.",
            "The companion robot session notes are shared with your child's oncology team.",
            "Future visits will include the same robot to build familiarity and trust.",
            "Report any anxiety or distress your child expresses about the robot to the care team.",
        ],
    },
    {  # Page 6: Humanoids (PEDIATRIC)
        "prep": [
            "Parents/guardians: show your child the humanoid robot introduction video sent by email.",
            "Pack a small snack and water bottle for after the session.",
            "Dress your child in comfortable clothes and non-slip shoes for walking activities.",
            "Arrive 10 minutes early so your child can see the robot through the window first.",
            "Inform the team of your child's preferred name and any communication needs.",
        ],
        "enter": [
            "Your child enters the activity room; the humanoid robot (about 1.5 m tall) stands inside.",
            "The robot waves and introduces itself by name in a child-friendly voice.",
            "The robot takes one slow step forward and stops, allowing your child to approach.",
            "Your child may hold the robot's hand if comfortable; the grip is very gentle.",
            "The robot kneels down to your child's eye level for face-to-face interaction.",
        ],
        "interact": [
            "The robot walks alongside your child at their pace (0.5--1.0 m/s maximum).",
            "Together they practice the route to the treatment room (approx. 3--5 minutes).",
            "The robot demonstrates deep breathing: 'Breathe in 1-2-3, out 1-2-3.' Your child copies.",
            "Your child can ask the robot questions; it answers in simple, reassuring language.",
            "Estimated session time: 15--25 minutes including walking and guided preparation.",
        ],
        "conclude": [
            "The robot walks your child back to the waiting area.",
            "It gives a gentle wave and says, 'Great job! See you next time.'",
            "The robot returns to its standby position in the activity room.",
            "A care team member brings your child to you with a session summary card.",
        ],
        "followup": [
            "Discuss with your child what they liked about the humanoid robot visit.",
            "The session summary is added to your child's electronic health record.",
            "The same humanoid robot will be used for consistency at future visits.",
            "Contact the pediatric oncology team if your child has questions or concerns.",
        ],
    },
    {  # Page 7: Radiotherapy Motion-Management / Tracking Robots
        "prep": [
            "Wear comfortable clothing without metal; a hospital gown may be provided.",
            "Do not eat a large meal within 1 hour of your session to reduce diaphragm movement.",
            "Practice the breathing pattern taught during your simulation: inhale 3 sec, exhale 3 sec.",
            "Avoid caffeine on the day of treatment, as it may increase involuntary movement.",
            "Bring your treatment schedule card for session verification.",
        ],
        "enter": [
            "Enter the treatment vault; the motion-tracking system has cameras mounted overhead.",
            "Lie on the treatment couch and position your arms as practiced in simulation.",
            "A technician places a small reflective marker block on your chest or abdomen.",
            "Breathe normally; the tracking cameras detect the markers and display your breathing trace.",
            "The system calibrates to your breathing pattern in approximately 1--2 minutes.",
        ],
        "interact": [
            "Follow the breathing guide on the ceiling-mounted screen: inhale when the bar rises.",
            "The system tracks your tumor position in real time; the beam activates only when aligned.",
            "You will hear clicking sounds as the beam gates on and off; this is expected.",
            "Remain as still as possible between breaths; the tracking tolerance is 2--3 mm.",
            "Estimated treatment delivery time: 10--20 minutes once tracking is calibrated.",
        ],
        "conclude": [
            "After the last beam, the tracking system deactivates and cameras power down.",
            "The technician removes the marker block from your chest.",
            "Sit up slowly and wait for the therapist to confirm session completion.",
            "Your breathing trace data is saved for comparison at future sessions.",
        ],
        "followup": [
            "Practice your breathing exercises at home for 5 minutes daily between sessions.",
            "Attend all scheduled treatment sessions to maintain the motion-management protocol.",
            "Report any new breathing difficulties or chest discomfort to your care team.",
            "Skin care for the treatment area follows the same guidelines as standard radiotherapy.",
        ],
    },
    {  # Page 8: Imaging Assistant Robots
        "prep": [
            "Drink 500 mL of water 1 hour before your appointment for ultrasound imaging clarity.",
            "Wear a two-piece outfit for easy access to the imaging area.",
            "Do not apply oils or lotion to the area being imaged.",
            "Bring previous imaging reports if this is a follow-up scan.",
            "Arrive 10 minutes early to complete a brief imaging questionnaire.",
        ],
        "enter": [
            "Enter the imaging room; the robotic ultrasound system is beside the exam bed.",
            "Lie on the exam bed and expose the area to be imaged.",
            "The robotic arm holds the ultrasound probe and will approach your body slowly.",
            "The technician applies ultrasound gel to the imaging area (it may feel cool).",
            "Place your hands behind your head or at your sides as instructed.",
        ],
        "interact": [
            "The robotic probe applies consistent, gentle pressure (1--3 N) on your skin.",
            "Breathe normally; the robot compensates for your breathing motion automatically.",
            "The robot scans methodically in a grid pattern for complete coverage of the target area.",
            "You may hear the motor of the robotic arm; this is normal operation.",
            "Estimated imaging time: 10--20 minutes depending on the area and scan complexity.",
        ],
        "conclude": [
            "The robotic arm retracts and the probe is lifted from your skin.",
            "The technician wipes excess gel from the imaged area with a warm cloth.",
            "Preliminary images appear on the display for immediate quality check.",
            "You may sit up and get dressed; there is no recovery period needed.",
        ],
        "followup": [
            "Imaging results are reviewed by a radiologist within 1--3 business days.",
            "Your oncologist will discuss findings at your next scheduled appointment.",
            "No special home care is needed after robotic ultrasound imaging.",
            "Future robotic imaging sessions maintain the same scan protocol for comparison.",
        ],
    },
    {  # Page 9: Steerable Needle / Needle-Steering Robots
        "prep": [
            "Fast for 6 hours before the procedure if sedation is planned.",
            "Inform the team of all current medications, especially blood thinners.",
            "An MRI compatibility screening form must be completed if MRI-guided.",
            "Remove all metal objects, credit cards, and electronic devices.",
            "Arrange transportation home; you should not drive after sedation.",
        ],
        "enter": [
            "Enter the interventional suite; the needle-steering robot is mounted to the imaging table.",
            "Lie on the table in the position indicated; pillows support your comfort.",
            "The area around the insertion site is cleaned and draped with sterile towels.",
            "A local anesthetic is injected at the entry point (brief 5-second sting).",
            "The robot's needle guide is positioned at the skin entry point.",
        ],
        "interact": [
            "The steerable needle curves through tissue along a pre-planned path to the target.",
            "You may feel deep pressure but should not feel sharp pain; speak up if you do.",
            "Real-time MRI or CT images confirm the needle tip position every 5--10 seconds.",
            "Hold still; the robot compensates for small movements but large shifts require re-planning.",
            "Estimated procedure time: 30--60 minutes for needle placement and treatment delivery.",
        ],
        "conclude": [
            "The steerable needle is slowly withdrawn along its insertion path.",
            "Pressure is held at the entry site for 5--10 minutes.",
            "A sterile dressing is applied and secured with medical tape.",
            "You are monitored in recovery for 30--60 minutes before discharge.",
        ],
        "followup": [
            "Rest at home for 24 hours; avoid bending or straining.",
            "Keep the dressing clean and dry for 48 hours.",
            "Take prescribed pain medication as directed (typically acetaminophen).",
            "Follow-up imaging is usually scheduled 1--2 weeks after the procedure.",
            "Call your care team for fever, worsening pain, or signs of infection.",
        ],
    },
    {  # Page 10: Rehabilitation Exoskeletons / Robotic Gait Trainers
        "prep": [
            "Wear fitted athletic clothing (leggings or joggers, T-shirt) and flat athletic shoes.",
            "Eat a moderate meal 1--2 hours before your session for sustained energy.",
            "Complete the pre-session mobility questionnaire sent to your phone.",
            "Bring a water bottle; hydration is important during physical rehabilitation.",
            "Inform the therapist of any new pain, swelling, or changes since your last session.",
        ],
        "enter": [
            "Enter the rehabilitation gym; the exoskeleton is hanging on a support frame between parallel bars.",
            "Stand with your back to the exoskeleton; a therapist secures the waist belt first.",
            "Thigh and shank straps are fastened snugly (you should feel firm but not tight pressure).",
            "Foot plates are adjusted to match your shoe size and secured with Velcro.",
            "Grip the parallel bars with both hands before the exoskeleton powers on.",
        ],
        "interact": [
            "The exoskeleton initiates a gentle standing motion; let it support your weight.",
            "Follow the audible 'step' cues: the exoskeleton moves one leg, then the other.",
            "Walk between the parallel bars at the robot's guided pace (0.2--0.5 m/s).",
            "If fatigued, say 'Pause'; the exoskeleton stops and supports you in a standing position.",
            "Estimated active walking time: 15--30 minutes with rest breaks as needed.",
        ],
        "conclude": [
            "The exoskeleton slowly returns you to a standing rest position.",
            "The therapist unbuckles straps in reverse order: feet, shank, thigh, waist.",
            "Sit in the provided chair for 5 minutes; drink water and rest.",
            "The therapist records your session metrics (steps, distance, gait symmetry).",
        ],
        "followup": [
            "Perform the prescribed home stretching exercises daily between sessions.",
            "Sessions are typically 2--3 times per week for 4--8 weeks.",
            "Track your fatigue, pain, and mobility in the patient diary provided.",
            "Progress metrics are reviewed at your monthly oncology rehabilitation appointment.",
        ],
    },
]


# ── Drawing helpers (reused from generate_illustrations.py) ──
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


def draw_head(ctx, cx, cy, radius):
    r = radius
    draw_circle(ctx, cx, cy, r)
    fill_white(ctx)
    eye_y = cy - r * 0.15
    eye_sep = r * 0.35
    for ex in [cx - eye_sep, cx + eye_sep]:
        ctx.arc(ex, eye_y, r * 0.07, 0, 2 * math.pi)
        ctx.fill()
    set_black(ctx, 1.5)
    ctx.move_to(cx, cy - r * 0.05)
    ctx.line_to(cx - r * 0.07, cy + r * 0.12)
    ctx.line_to(cx + r * 0.07, cy + r * 0.12)
    ctx.stroke()
    ctx.arc(cx, cy + r * 0.22, r * 0.18, 0.15, math.pi - 0.15)
    ctx.stroke()
    set_black(ctx, 2.0)


def draw_hair_variant(ctx, cx, cy, r, variant):
    """Draw hair based on variant index for diversity."""
    if variant == 0:  # Short
        ctx.arc(cx, cy, r * 1.05, -math.pi, 0)
        ctx.stroke()
        for angle in [-2.5, -2.0, -1.5, -1.0, -0.5]:
            x1 = cx + r * 0.95 * math.cos(angle)
            y1 = cy + r * 0.95 * math.sin(angle)
            x2 = cx + r * 1.12 * math.cos(angle)
            y2 = cy + r * 1.12 * math.sin(angle)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
        ctx.stroke()
    elif variant == 1:  # Long
        ctx.move_to(cx - r * 1.05, cy - r * 0.2)
        ctx.curve_to(cx - r * 1.15, cy + r * 0.5, cx - r * 1.05, cy + r * 1.0, cx - r * 0.6, cy + r * 1.3)
        ctx.stroke()
        ctx.move_to(cx + r * 1.05, cy - r * 0.2)
        ctx.curve_to(cx + r * 1.15, cy + r * 0.5, cx + r * 1.05, cy + r * 1.0, cx + r * 0.6, cy + r * 1.3)
        ctx.stroke()
        ctx.arc(cx, cy - r * 0.15, r * 1.06, -math.pi, 0)
        ctx.stroke()
    elif variant == 2:  # Curly
        set_black(ctx, 1.5)
        for angle_deg in range(0, 360, 14):
            a = math.radians(angle_deg)
            bx = cx + r * 1.12 * math.cos(a)
            by = cy + r * 1.12 * math.sin(a)
            ctx.arc(bx, by, r * 0.1, 0, 2 * math.pi)
            ctx.stroke()
        set_black(ctx, 2.0)
    elif variant == 3:  # Bald/stubble
        set_black(ctx, 1.0)
        for angle_deg in range(-160, -20, 18):
            a = math.radians(angle_deg)
            bx = cx + r * 0.92 * math.cos(a)
            by = cy + r * 0.92 * math.sin(a)
            ctx.arc(bx, by, 1.0, 0, 2 * math.pi)
            ctx.fill()
        set_black(ctx, 2.0)
    elif variant == 4:  # Headscarf
        ctx.arc(cx, cy - r * 0.1, r * 1.1, -math.pi + 0.3, -0.3)
        ctx.line_to(cx + r * 0.85, cy + r * 0.85)
        ctx.curve_to(cx + r * 0.4, cy + r * 1.2, cx - r * 0.4, cy + r * 1.2, cx - r * 0.85, cy + r * 0.85)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()
    elif variant == 5:  # Short textured
        for angle_deg in range(-170, -10, 12):
            a = math.radians(angle_deg)
            x1 = cx + r * 0.95 * math.cos(a)
            y1 = cy + r * 0.95 * math.sin(a)
            x2 = cx + r * 1.15 * math.cos(a + 0.1)
            y2 = cy + r * 1.15 * math.sin(a + 0.1)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
        ctx.stroke()
    elif variant == 6:  # Ponytail
        ctx.arc(cx, cy - r * 0.15, r * 1.06, -math.pi, 0)
        ctx.stroke()
        ctx.move_to(cx, cy - r * 1.06)
        ctx.curve_to(cx + r * 0.5, cy - r * 1.5, cx + r * 1.0, cy - r * 0.8, cx + r * 0.8, cy)
        ctx.stroke()
    elif variant == 7:  # Child pigtails
        ctx.arc(cx, cy - r * 0.15, r * 1.05, -math.pi, 0)
        ctx.stroke()
        for side in [-1, 1]:
            ctx.move_to(cx + side * r * 0.85, cy - r * 0.5)
            ctx.curve_to(
                cx + side * r * 1.3,
                cy - r * 0.3,
                cx + side * r * 1.2,
                cy + r * 0.4,
                cx + side * r * 0.9,
                cy + r * 0.7,
            )
            ctx.stroke()
            ctx.arc(cx + side * r * 0.85, cy - r * 0.5, r * 0.06, 0, 2 * math.pi)
            ctx.stroke()
    elif variant == 8:  # Child short messy
        set_black(ctx, 1.5)
        for angle_deg in range(-170, -10, 16):
            a = math.radians(angle_deg)
            x1 = cx + r * 0.95 * math.cos(a)
            y1 = cy + r * 0.95 * math.sin(a)
            x2 = cx + r * 1.18 * math.cos(a + 0.12)
            y2 = cy + r * 1.18 * math.sin(a + 0.12)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
        ctx.stroke()
        set_black(ctx, 2.0)


# Hair variant assignment per page (diverse)
HAIR_VARIANTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]


def draw_patient_body_small(ctx, cx, cy, body_h, is_child=False, gown=True):
    """Draw a small patient body for the PDF illustration."""
    sw = body_h * 0.5 if not is_child else body_h * 0.42
    if gown:
        ctx.move_to(cx - sw / 2, cy)
        ctx.line_to(cx - sw / 2 - 8, cy + body_h * 0.6)
        ctx.line_to(cx + sw / 2 + 8, cy + body_h * 0.6)
        ctx.line_to(cx + sw / 2, cy)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()
        ctx.move_to(cx - 8, cy)
        ctx.line_to(cx, cy + 15)
        ctx.line_to(cx + 8, cy)
        ctx.stroke()
    else:
        ctx.move_to(cx - sw / 2, cy)
        ctx.line_to(cx - sw / 2 - 5, cy + body_h * 0.6)
        ctx.line_to(cx + sw / 2 + 5, cy + body_h * 0.6)
        ctx.line_to(cx + sw / 2, cy)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()


def draw_arm_reaching_small(ctx, sx, sy, hx, hy):
    mx = (sx + hx) / 2
    my = min(sy, hy) - 8
    ctx.move_to(sx, sy)
    ctx.curve_to(mx, my, hx - 5, hy - 10, hx, hy)
    ctx.stroke()
    ctx.arc(hx, hy, 4, 0, 2 * math.pi)
    fill_white(ctx)


def draw_arm_down_small(ctx, sx, sy, length):
    ctx.move_to(sx, sy)
    ctx.line_to(sx - 5, sy + length)
    ctx.stroke()
    ctx.arc(sx - 5, sy + length, 3.5, 0, 2 * math.pi)
    fill_white(ctx)


# ── Compact robot drawing functions for PDF ──────────────────
def draw_surgical_robot_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 25, ry + 35, 50, 80, 4)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 16, ry + 42, 32, 18, 2)
    ctx.stroke()
    ctx.move_to(rx, ry + 35)
    ctx.line_to(rx, ry - 15)
    ctx.stroke()
    for ang, al in [(-35, 50), (0, 55), (35, 50)]:
        a = math.radians(ang - 90)
        ax = rx + al * math.cos(a) * 0.7
        ay = ry - 15 + al * math.sin(a) * 0.7
        ctx.move_to(rx, ry - 15)
        ctx.line_to(ax, ay)
        ctx.stroke()
        ctx.arc(ax, ay, 2.5, 0, 2 * math.pi)
        fill_white(ctx)
        ex = ax + 20 * math.cos(a + 0.3)
        ey = ay + 20 * math.sin(a + 0.3)
        ctx.move_to(ax, ay)
        ctx.line_to(ex, ey)
        ctx.stroke()
    for wx in [rx - 18, rx + 18]:
        ctx.arc(wx, ry + 118, 4, 0, 2 * math.pi)
        fill_white(ctx)


def draw_cobot_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 22, ry + 65, 44, 35, 5)
    fill_white(ctx)
    for wx in [rx - 15, rx + 15]:
        ctx.arc(wx, ry + 103, 4, 0, 2 * math.pi)
        fill_white(ctx)
    ctx.move_to(rx - 10, ry + 65)
    ctx.line_to(rx - 10, ry + 52)
    ctx.line_to(rx + 10, ry + 52)
    ctx.line_to(rx + 10, ry + 65)
    ctx.stroke()
    joints = [(rx, ry + 52), (rx + 15, ry + 30), (rx + 25, ry + 10), (rx + 15, ry - 10), (rx, ry - 20)]
    set_black(ctx, 2.0)
    for i in range(len(joints) - 1):
        ctx.move_to(joints[i][0], joints[i][1])
        ctx.line_to(joints[i + 1][0], joints[i + 1][1])
        ctx.stroke()
    set_black(ctx, 2.0)
    for jx, jy in joints:
        ctx.arc(jx, jy, 3, 0, 2 * math.pi)
        fill_white(ctx)
    ex, ey = joints[-1]
    ctx.move_to(ex, ey)
    ctx.line_to(ex - 7, ey - 10)
    ctx.stroke()
    ctx.move_to(ex, ey)
    ctx.line_to(ex + 3, ey - 11)
    ctx.stroke()


def draw_rt_positioning_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 35, ry + 75, 70, 15, 3)
    fill_white(ctx)
    ctx.move_to(rx - 8, ry + 75)
    ctx.line_to(rx - 10, ry + 30)
    ctx.line_to(rx + 10, ry + 30)
    ctx.line_to(rx + 8, ry + 75)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.0)
    ctx.stroke()
    draw_rounded_rect(ctx, rx - 50, ry + 18, 100, 14, 3)
    fill_white(ctx)
    set_black(ctx, 2.0)
    ctx.arc(rx, ry - 10, 42, -math.pi + 0.3, -0.3)
    ctx.stroke()
    ctx.set_dash([2, 2])
    ctx.move_to(rx, ry - 10)
    ctx.line_to(rx, ry + 18)
    ctx.stroke()
    ctx.set_dash([])


def draw_needle_placement_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 22, ry + 50, 44, 55, 4)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 15, ry + 56, 30, 20, 2)
    ctx.stroke()
    joints = [(rx, ry + 50), (rx + 15, ry + 28), (rx + 22, ry + 5), (rx + 12, ry - 12)]
    set_black(ctx, 2.0)
    for i in range(len(joints) - 1):
        ctx.move_to(joints[i][0], joints[i][1])
        ctx.line_to(joints[i + 1][0], joints[i + 1][1])
        ctx.stroke()
    for jx, jy in joints:
        ctx.arc(jx, jy, 2.5, 0, 2 * math.pi)
        fill_white(ctx)
    ex, ey = joints[-1]
    draw_rounded_rect(ctx, ex - 7, ey - 7, 14, 14, 2)
    fill_white(ctx)
    ctx.move_to(ex, ey)
    ctx.line_to(ex - 12, ey + 25)
    ctx.stroke()
    for wx in [rx - 15, rx + 15]:
        ctx.arc(wx, ry + 108, 4, 0, 2 * math.pi)
        fill_white(ctx)


def draw_companion_small(ctx, rx, ry):
    ctx.arc(rx, ry - 10, 25, 0, 2 * math.pi)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.0)
    ctx.stroke()
    for ex in [rx - 9, rx + 9]:
        ctx.arc(ex, ry - 14, 6, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 1.5)
        ctx.stroke()
        ctx.arc(ex + 1, ry - 15, 2.5, 0, 2 * math.pi)
        ctx.fill()
    set_black(ctx, 2.0)
    ctx.arc(rx, ry + 1, 8, 0.2, math.pi - 0.2)
    ctx.stroke()
    draw_rounded_rect(ctx, rx - 20, ry + 20, 40, 50, 10)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 12, ry + 30, 24, 18, 3)
    ctx.stroke()
    for side in [-1, 1]:
        ctx.move_to(rx + side * 20, ry + 28)
        ctx.curve_to(rx + side * 32, ry + 35, rx + side * 30, ry + 55, rx + side * 18, ry + 60)
        ctx.stroke()
        ctx.arc(rx + side * 18, ry + 60, 4, 0, 2 * math.pi)
        fill_white(ctx)
    ctx.move_to(rx - 15, ry + 70)
    ctx.line_to(rx - 18, ry + 85)
    ctx.line_to(rx + 18, ry + 85)
    ctx.line_to(rx + 15, ry + 70)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.0)
    ctx.stroke()


def draw_humanoid_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 14, ry - 32, 28, 24, 6)
    fill_white(ctx)
    for ex in [rx - 6, rx + 6]:
        draw_rounded_rect(ctx, ex - 4, ry - 27, 8, 4, 1.5)
        ctx.stroke()
    ctx.move_to(rx - 5, ry - 15)
    ctx.line_to(rx + 5, ry - 15)
    ctx.stroke()
    ctx.move_to(rx - 18, ry - 5)
    ctx.line_to(rx - 16, ry + 38)
    ctx.line_to(rx + 16, ry + 38)
    ctx.line_to(rx + 18, ry - 5)
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    set_black(ctx, 2.0)
    ctx.stroke()
    for side in [-1, 1]:
        sx = rx + side * 20
        ctx.arc(sx, ry - 2, 4, 0, 2 * math.pi)
        fill_white(ctx)
        ebx = sx + side * 12
        eby = ry + 18
        ctx.move_to(sx, ry - 2)
        ctx.line_to(ebx, eby)
        ctx.stroke()
        ctx.arc(ebx, eby, 3, 0, 2 * math.pi)
        fill_white(ctx)
        hndx = ebx + side * 5
        hndy = eby + 20
        ctx.move_to(ebx, eby)
        ctx.line_to(hndx, hndy)
        ctx.stroke()
        ctx.arc(hndx, hndy, 4, 0, 2 * math.pi)
        fill_white(ctx)
    for side in [-1, 1]:
        hpx = rx + side * 10
        kx = hpx + side * 2
        ky = ry + 60
        ctx.move_to(hpx, ry + 38)
        ctx.line_to(kx, ky)
        ctx.stroke()
        ctx.arc(kx, ky, 3, 0, 2 * math.pi)
        fill_white(ctx)
        fx = kx
        fy = ky + 25
        ctx.move_to(kx, ky)
        ctx.line_to(fx, fy)
        ctx.stroke()
        ctx.move_to(fx - 5, fy)
        ctx.line_to(fx + 8, fy)
        ctx.line_to(fx + 8, fy + 4)
        ctx.line_to(fx - 5, fy + 4)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()


def draw_motion_tracking_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 25, ry + 28, 50, 65, 4)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 18, ry + 35, 36, 22, 2)
    ctx.stroke()
    set_black(ctx, 1.0)
    pts = []
    for i in range(12):
        px = rx - 15 + i * 2.5
        py = ry + 46 + 5 * math.sin(i * 0.8)
        pts.append((px, py))
    ctx.move_to(pts[0][0], pts[0][1])
    for px, py in pts[1:]:
        ctx.line_to(px, py)
    ctx.stroke()
    set_black(ctx, 2.0)
    ctx.move_to(rx, ry + 28)
    ctx.line_to(rx, ry + 10)
    ctx.stroke()
    ctx.move_to(rx - 28, ry + 10)
    ctx.line_to(rx + 28, ry + 10)
    ctx.stroke()
    for dx in [-20, 0, 20]:
        draw_rounded_rect(ctx, rx + dx - 6, ry - 2, 12, 12, 2)
        fill_white(ctx)
        ctx.arc(rx + dx, ry + 4, 3, 0, 2 * math.pi)
        ctx.fill()
    ctx.set_dash([2, 2])
    for dx in [-20, 0, 20]:
        ctx.move_to(rx + dx, ry + 10)
        ctx.line_to(rx + dx * 0.3, ry + 24)
        ctx.stroke()
    ctx.set_dash([])
    for wx in [rx - 18, rx + 18]:
        ctx.arc(wx, ry + 96, 4, 0, 2 * math.pi)
        fill_white(ctx)


def draw_imaging_assistant_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 20, ry + 52, 40, 42, 4)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 16, ry - 25, 32, 25, 3)
    fill_white(ctx)
    set_black(ctx, 1.0)
    ctx.arc(rx, ry - 14, 8, 0, 2 * math.pi)
    ctx.stroke()
    set_black(ctx, 2.0)
    ctx.move_to(rx, ry)
    ctx.curve_to(rx + 5, ry + 12, rx - 5, ry + 25, rx, ry + 35)
    ctx.stroke()
    ctx.move_to(rx, ry + 35)
    ctx.line_to(rx - 15, ry + 20)
    ctx.stroke()
    ctx.arc(rx - 15, ry + 20, 2.5, 0, 2 * math.pi)
    fill_white(ctx)
    ctx.move_to(rx - 15, ry + 20)
    ctx.line_to(rx - 28, ry + 12)
    ctx.stroke()
    draw_rounded_rect(ctx, rx - 38, ry + 5, 15, 15, 2)
    fill_white(ctx)
    ctx.move_to(rx - 30, ry + 20)
    ctx.line_to(rx - 30, ry + 35)
    ctx.stroke()
    ctx.arc(rx - 30, ry + 38, 4, 0, math.pi)
    ctx.stroke()
    for wx in [rx - 12, rx + 12]:
        ctx.arc(wx, ry + 97, 4, 0, 2 * math.pi)
        fill_white(ctx)


def draw_steerable_needle_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 22, ry + 30, 44, 35, 3)
    fill_white(ctx)
    set_black(ctx, 2.0)
    ctx.move_to(rx + 5, ry + 30)
    ctx.line_to(rx + 5, ry + 5)
    ctx.stroke()
    ctx.move_to(rx - 5, ry + 30)
    ctx.line_to(rx - 5, ry + 5)
    ctx.stroke()
    ctx.move_to(rx, ry + 5)
    ctx.curve_to(rx - 3, ry - 10, rx - 8, ry - 25, rx - 12, ry - 38)
    ctx.stroke()
    ctx.move_to(rx - 12, ry - 38)
    ctx.line_to(rx - 14, ry - 44)
    ctx.line_to(rx - 10, ry - 43)
    ctx.close_path()
    ctx.fill()
    ctx.set_dash([2, 2])
    ctx.arc(rx - 14, ry - 50, 7, 0, 2 * math.pi)
    ctx.stroke()
    ctx.set_dash([])
    draw_rounded_rect(ctx, rx - 18, ry + 35, 36, 16, 2)
    ctx.stroke()
    for mdx in [-8, 0, 8]:
        ctx.arc(rx + mdx, ry + 43, 2.5, 0, 2 * math.pi)
        fill_white(ctx)


def draw_exoskeleton_small(ctx, rx, ry):
    draw_rounded_rect(ctx, rx - 25, ry - 18, 50, 14, 3)
    fill_white(ctx)
    draw_rounded_rect(ctx, rx - 12, ry - 28, 24, 18, 3)
    fill_white(ctx)
    set_black(ctx, 2.5)
    ctx.arc(rx, ry - 11, 23, -0.3, math.pi + 0.3)
    ctx.stroke()
    set_black(ctx, 2.0)
    for side in [-1, 1]:
        hpx = rx + side * 18
        ctx.arc(hpx, ry - 2, 5, 0, 2 * math.pi)
        fill_white(ctx)
        kx = hpx + side * 2
        ky = ry + 28
        set_black(ctx, 2.0)
        ctx.move_to(hpx, ry - 2)
        ctx.line_to(kx, ky)
        ctx.stroke()
        ctx.arc(kx, ky, 5, 0, 2 * math.pi)
        fill_white(ctx)
        ax = kx
        ay = ky + 25
        ctx.move_to(kx, ky)
        ctx.line_to(ax, ay)
        ctx.stroke()
        ctx.arc(ax, ay, 3.5, 0, 2 * math.pi)
        fill_white(ctx)
        ctx.move_to(ax - 7, ay + 3)
        ctx.line_to(ax + 9, ay + 3)
        ctx.line_to(ax + 10, ay + 7)
        ctx.line_to(ax - 7, ay + 7)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 2.0)
        ctx.stroke()
    for side in [-1, 1]:
        bar_x = rx + side * 42
        ctx.move_to(bar_x, ry - 30)
        ctx.line_to(bar_x, ry + 60)
        ctx.stroke()


SMALL_ROBOT_DRAWERS = [
    draw_surgical_robot_small,
    draw_cobot_small,
    draw_rt_positioning_small,
    draw_needle_placement_small,
    draw_companion_small,
    draw_humanoid_small,
    draw_motion_tracking_small,
    draw_imaging_assistant_small,
    draw_steerable_needle_small,
    draw_exoskeleton_small,
]


# ── ISO symbol helpers (small) ───────────────────────────────
def draw_iso_symbol_small(ctx, cx, cy, page_idx):
    """Draw a small ISO symbol appropriate for the page."""
    size = 14
    if page_idx in {0, 3, 8}:  # Caution
        h = size * 0.866
        ctx.move_to(cx, cy - h * 0.6)
        ctx.line_to(cx - size / 2, cy + h * 0.4)
        ctx.line_to(cx + size / 2, cy + h * 0.4)
        ctx.close_path()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        set_black(ctx, 1.5)
        ctx.stroke()
        ctx.move_to(cx, cy - h * 0.15)
        ctx.line_to(cx, cy + h * 0.1)
        ctx.stroke()
        ctx.arc(cx, cy + h * 0.2, 1, 0, 2 * math.pi)
        ctx.fill()
    elif page_idx in {2, 6}:  # Radiation
        r_inner = size * 0.15
        r_outer = size * 0.4
        set_black(ctx, 1.0)
        for i in range(3):
            a_start = math.radians(i * 120 - 90 + 10)
            a_end = math.radians(i * 120 - 90 + 110)
            ctx.arc(cx, cy, r_outer, a_start, a_end)
            ctx.arc_negative(cx, cy, r_inner, a_end, a_start)
            ctx.close_path()
            ctx.fill()
        ctx.arc(cx, cy, r_inner * 0.5, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()
        set_black(ctx, 2.0)
    else:  # Keep still
        set_black(ctx, 1.0)
        hr = size * 0.1
        ctx.arc(cx, cy - size * 0.25, hr, 0, 2 * math.pi)
        ctx.stroke()
        ctx.move_to(cx, cy - size * 0.15)
        ctx.line_to(cx, cy + size * 0.1)
        ctx.stroke()
        ctx.move_to(cx - size * 0.15, cy - size * 0.02)
        ctx.line_to(cx + size * 0.15, cy - size * 0.02)
        ctx.stroke()
        ctx.move_to(cx, cy + size * 0.1)
        ctx.line_to(cx - size * 0.1, cy + size * 0.28)
        ctx.stroke()
        ctx.move_to(cx, cy + size * 0.1)
        ctx.line_to(cx + size * 0.1, cy + size * 0.28)
        ctx.stroke()
        set_black(ctx, 2.0)


# ── Text wrapping helper ────────────────────────────────────
def wrap_text(ctx, text, max_width):
    """Word-wrap text to fit within max_width, returning list of lines."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        extents = ctx.text_extents(test_line)
        if extents.width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


# ── Main page drawing function ───────────────────────────────
def draw_page(surface, page_idx):
    """Draw a complete page on the given PDF surface."""
    ctx = cairo.Context(surface)
    is_child = page_idx in PEDIATRIC_PAGES
    robot_type = ROBOT_TYPES[page_idx]

    # White background
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, PAGE_W, PAGE_H)
    ctx.fill()

    set_black(ctx, 0.5)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # ── (a) Header bar ──
    header_y = MARGIN_TOP
    set_black(ctx, 0.8)
    ctx.move_to(MARGIN_LEFT, header_y + 12)
    ctx.line_to(PAGE_W - MARGIN_RIGHT, header_y + 12)
    ctx.stroke()

    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(6.5)
    header_text = (
        "Kevin Kawchak, CEO ChemicalQDevice, https://orcid.org/0009-0007-5457-8667, kevink@chemicalqdevice.com"
    )
    ctx.move_to(MARGIN_LEFT, header_y + 8)
    ctx.show_text(header_text)

    # ── (b) Title ──
    title_y = header_y + 30
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(13)
    title = f"Patient-Robot Instructions: Physical AI Oncology Trials - {robot_type}"
    # Check if title fits
    extents = ctx.text_extents(title)
    if extents.width > CONTENT_W:
        ctx.set_font_size(10.5)
        extents = ctx.text_extents(title)
    if extents.width > CONTENT_W:
        ctx.set_font_size(9)
        extents = ctx.text_extents(title)
    ctx.move_to((PAGE_W - extents.width) / 2, title_y)
    ctx.show_text(title)

    # Title underline
    set_black(ctx, 0.5)
    ctx.move_to(MARGIN_LEFT, title_y + 5)
    ctx.line_to(PAGE_W - MARGIN_RIGHT, title_y + 5)
    ctx.stroke()

    # ── (c) Illustration ──
    illust_top = title_y + 14
    illust_h = 200
    illust_center_x = PAGE_W / 2
    illust_center_y = illust_top + illust_h / 2

    # Light border for illustration area
    set_black(ctx, 0.3)
    draw_rounded_rect(ctx, MARGIN_LEFT + 10, illust_top, CONTENT_W - 20, illust_h, 6)
    ctx.stroke()

    # Scene label
    ctx.select_font_face("serif", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(7)
    scene = "Pediatric Oncology Trial Setting" if is_child else "Adult Oncology Trial Setting"
    ext = ctx.text_extents(scene)
    ctx.move_to((PAGE_W - ext.width) / 2, illust_top + 12)
    ctx.show_text(scene)

    # Floor line
    set_black(ctx, 0.3)
    floor_y = illust_top + illust_h - 15
    ctx.set_dash([3, 2])
    ctx.move_to(MARGIN_LEFT + 20, floor_y)
    ctx.line_to(PAGE_W - MARGIN_RIGHT - 20, floor_y)
    ctx.stroke()
    ctx.set_dash([])

    # Draw robot (right side)
    robot_x = illust_center_x + 65
    robot_y = illust_center_y + 5
    set_black(ctx, 1.5)
    ctx.save()
    sf = 1.35 if not is_child else 1.1
    ctx.translate(robot_x, robot_y)
    ctx.scale(sf, sf)
    ctx.translate(-robot_x, -robot_y)
    SMALL_ROBOT_DRAWERS[page_idx](ctx, robot_x, robot_y)
    ctx.restore()

    # Draw patient (left side)
    patient_x = illust_center_x - 80
    patient_y = illust_center_y - 30 if not is_child else illust_center_y - 20
    head_r = 16 if not is_child else 13
    body_h = 80 if not is_child else 60

    set_black(ctx, 1.5)
    draw_head(ctx, patient_x, patient_y, head_r)
    draw_hair_variant(ctx, patient_x, patient_y, head_r, HAIR_VARIANTS[page_idx])

    shoulder_y = patient_y + head_r + 3
    draw_patient_body_small(ctx, patient_x, shoulder_y, body_h, is_child)

    # Arm reaching toward robot
    s_x = patient_x + (15 if not is_child else 10)
    r_x = robot_x - 20
    r_y = patient_y + head_r + 25
    draw_arm_reaching_small(ctx, s_x, shoulder_y + 5, r_x, r_y)

    # Other arm
    other_s = patient_x - (15 if not is_child else 10)
    draw_arm_down_small(ctx, other_s, shoulder_y + 5, body_h * 0.35)

    # ISO symbol in illustration
    draw_iso_symbol_small(ctx, PAGE_W - MARGIN_RIGHT - 30, illust_top + illust_h - 28, page_idx)

    # Robot label under illustration
    set_black(ctx, 0.8)
    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(8.5)
    lbl = robot_type
    ext = ctx.text_extents(lbl)
    ctx.move_to((PAGE_W - ext.width) / 2, illust_top + illust_h - 3)
    ctx.show_text(lbl)

    # ── (d) Instructions ──
    inst_top = illust_top + illust_h + 8
    inst = INSTRUCTIONS[page_idx]
    y_cursor = inst_top

    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    text_width = CONTENT_W - 20
    left_x = MARGIN_LEFT + 10

    sections = [
        ("1. Preparation at Home", inst["prep"]),
        ("2. Entering the Room", inst["enter"]),
        ("3. During the Interaction", inst["interact"]),
        ("4. Concluding the Session", inst["conclude"]),
        ("5. At Home & Follow-Up", inst["followup"]),
    ]

    for sec_title, items in sections:
        # Section header
        ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(8.5)
        set_black(ctx, 0.8)
        ctx.move_to(left_x, y_cursor)
        ctx.show_text(sec_title)
        y_cursor += 2

        # Underline
        ext = ctx.text_extents(sec_title)
        set_black(ctx, 0.3)
        ctx.move_to(left_x, y_cursor)
        ctx.line_to(left_x + ext.width, y_cursor)
        ctx.stroke()
        y_cursor += 8

        # Bullet items
        ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(7.2)
        set_black(ctx, 0.8)

        for i, item in enumerate(items):
            bullet_x = left_x + 8
            text_x = left_x + 18

            # Numbered bullet
            ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(7.2)
            num_str = f"{i + 1}."
            ctx.move_to(bullet_x, y_cursor)
            ctx.show_text(num_str)

            ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(7.2)

            # Wrap text
            lines = wrap_text(ctx, item, text_width - 20)
            for line in lines:
                ctx.move_to(text_x, y_cursor)
                ctx.show_text(line)
                y_cursor += 9
            y_cursor += 1

        y_cursor += 4

    # ── (e) Footer bar ──
    footer_y = PAGE_H - MARGIN_BOTTOM - 5
    set_black(ctx, 0.8)
    ctx.move_to(MARGIN_LEFT, footer_y - 10)
    ctx.line_to(PAGE_W - MARGIN_RIGHT, footer_y - 10)
    ctx.stroke()

    ctx.select_font_face("serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(6)

    # Left: date + DOI
    footer_left = "February 28, 2026  |  10.5281/zenodo.18810541 (doi.org/10.5281/zenodo.18810541)"
    ctx.move_to(MARGIN_LEFT, footer_y)
    ctx.show_text(footer_left)

    # Right: model + page
    footer_right = f"Claude Code Opus 4.6  |  Page {page_idx + 1} of 10"
    ext = ctx.text_extents(footer_right)
    ctx.move_to(PAGE_W - MARGIN_RIGHT - ext.width, footer_y)
    ctx.show_text(footer_right)

    # Sources line
    ctx.set_font_size(5.5)
    sources = PAGE_SOURCES[page_idx]
    source_line = f"Sources: {sources}"
    ctx.move_to(MARGIN_LEFT, footer_y + 9)
    ctx.show_text(source_line)


def generate_combined_pdf():
    """Generate the complete 10-page PDF."""
    output_path = os.path.join(BASE_DIR, "paper", "Patient-Robot Instructions: Physical AI Oncology Trials.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    surface = cairo.PDFSurface(output_path, PAGE_W, PAGE_H)

    for idx in range(10):
        print(f"Drawing page {idx + 1}/10: {ROBOT_TYPES[idx]}")
        draw_page(surface, idx)
        surface.show_page()

    surface.finish()
    print(f"\nCombined PDF saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_combined_pdf()
