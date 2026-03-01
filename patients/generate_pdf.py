"""Generate Patient-Robot Instructions PDF for Physical AI Oncology Trials (v1.9.1).

Produces a 10-page PDF where each page is a self-contained instruction sheet
for one robot type. Uses reportlab for PDF generation and Pillow for image
handling. Outputs three versions: full-size, 10 MB target, and 5 MB target.

License: MIT
Author: Kevin Kawchak, CEO ChemicalQDevice
Date: March 1, 2026
AI: Claude Code Opus 4.6
"""

import os
import re
import shutil

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4  # 210 x 297 mm
MARGIN_L = 20 * mm
MARGIN_R = 20 * mm
MARGIN_T = 22 * mm
MARGIN_B = 18 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
PAPER_DIR = os.path.join(os.path.dirname(__file__), "paper")

DOI = "10.5281/zenodo.18810541"
DOI_URL = "https://doi.org/10.5281/zenodo.18810541"
AUTHOR_LINE = "Kevin Kawchak, CEO ChemicalQDevice, https://orcid.org/0009-0007-5457-8667, kevink@chemicalqdevice.com"
DATE_STR = "March 1, 2026"

# ---------------------------------------------------------------------------
# Page data — robot types, titles, instructions
# ---------------------------------------------------------------------------
PAGES = [
    {
        "num": 1,
        "robot_full": "Surgical Robots",
        "robot_abbr": "Surgical Robots",
        "image": "1.png",
        "cancer": "prostate cancer",
        "intro": (
            "This surgical robot assists your surgeon in performing a minimally "
            "invasive robotic prostatectomy through small incisions using "
            "precise multi-arm instruments."
        ),
        "steps": [
            (
                "Lie face-up on the operating table, place both arms flat on "
                "the padded arm rests with palms up, and remain still while "
                "anesthesia is administered over approximately 2 minutes."
            ),
            (
                "You will be under general anesthesia for the entire 90\u2013180 "
                "minute procedure; the robot\u2019s 4 arms operate through "
                "8\u201312 mm incisions while the surgeon controls it from a "
                "nearby console."
            ),
            (
                "After the robot arms retract, you are moved to recovery where "
                "you will awaken in 15\u201330 minutes; remain lying down until "
                "a nurse confirms your vitals."
            ),
        ],
        "sources": (
            '<link href="https://www.intuitivesurgical.com/products/da-vinci">'
            "Intuitive Surgical</link> "
            "[1]; Sheetz et al. 2020 [2]; Moglia et al. 2021 [3]"
        ),
    },
    {
        "num": 2,
        "robot_full": "Cobots (Collaborative Robots)",
        "robot_abbr": "Cobots",
        "image": "2.png",
        "cancer": "breast cancer",
        "intro": (
            "This collaborative robot arm gently positions your arm and "
            "assists with automated tissue sample collection for your "
            "breast cancer biopsy analysis."
        ),
        "steps": [
            (
                "Sit in the adjustable exam chair, rest your arm on the "
                "padded support with your palm facing down, and keep your "
                "hand relaxed and fingers open for approximately 1 minute."
            ),
            (
                "The cobot moves at a maximum of 250 mm/s with force-limited "
                "contact; hold still for 10\u201325 minutes while it "
                "repositions your arm 2\u20133 times and collects samples, "
                "saying \u201cStop\u201d if you feel discomfort."
            ),
            (
                "The cobot retracts to its home position; remain seated for "
                "2 minutes while results are reviewed, then receive a "
                "bandage at the collection site before leaving."
            ),
        ],
        "sources": (
            '<link href="https://franka.de/products/production-robot">'
            "Franka Robotics</link> "
            "[4]; Lamon et al. 2023 [5]; Haddadin & Croft 2016 [6]"
        ),
    },
    {
        "num": 3,
        "robot_full": "Radiotherapy Patient-Positioning Robots",
        "robot_abbr": "RT Positioning Robots",
        "image": "3.png",
        "cancer": "lung cancer",
        "intro": (
            "This robotic treatment couch precisely positions your body "
            "using six degrees of freedom to align with the radiation beam "
            "for your lung tumor treatment."
        ),
        "steps": [
            (
                "Lie on the robotic couch in the position practiced during "
                "simulation, place your arms in the overhead cradle, and "
                "allow the immobilization mask to be fitted over 2\u20133 "
                "minutes."
            ),
            (
                "The couch shifts in 1\u20133 mm increments to align you "
                "with the beam; breathe normally and remain completely still "
                "during the 15\u201330 minute session, keeping your hands "
                "above your head."
            ),
            (
                "When the beam stops, the couch lowers to exit height; "
                "wait for the immobilization device to be removed, then sit "
                "up slowly using both hands for support."
            ),
        ],
        "sources": (
            '<link href="https://www.accuray.com/treatment-delivery/'
            'cyberknife-system/">Accuray</link> '
            "[7]; Hoisak & Pawlicki 2022 [8]; Verellen et al. 2007 [9]"
        ),
    },
    {
        "num": 4,
        "robot_full": "Robotic Needle-Placement Systems",
        "robot_abbr": "Needle-Placement Robots",
        "image": "4.png",
        "cancer": "liver cancer",
        "intro": (
            "This robotic needle-guidance system uses real-time CT imaging "
            "to place a biopsy needle with sub-millimeter accuracy into "
            "your liver lesion."
        ),
        "steps": [
            (
                "Lie on the imaging table face-up with both hands at your "
                "sides, hold still while local anesthetic numbs the "
                "insertion site for 1\u20132 minutes, and keep your "
                "breathing shallow."
            ),
            (
                "The robot aligns and inserts the needle over 1\u20133 "
                "minutes; hold your breath for 5\u201310 seconds when "
                "prompted for image capture during the 20\u201345 minute "
                "total procedure."
            ),
            (
                "After the needle is withdrawn, press gently on the gauze "
                "placed over the site for 3\u20135 minutes, then remain "
                "lying down for 15\u201330 minutes of monitoring."
            ),
        ],
        "sources": (
            "Walsh et al. 2021 [10]; Li et al. 2023 [11]; "
            '<link href="https://www.iso.org/standard/77326.html">'
            "ISO 15223-1</link> [12]"
        ),
    },
    {
        "num": 5,
        "robot_full": "Social Companion Robots",
        "robot_abbr": "Companion Robots",
        "image": "5.png",
        "cancer": "pediatric leukemia",
        "intro": (
            "This friendly companion robot will talk and play interactive "
            "games with you to help you feel comfortable before your "
            "leukemia treatment today."
        ),
        "steps": [
            (
                "Walk into the room and wave or say \u201cHello\u201d to "
                "the 1.2-meter tall robot; it will greet you by name and "
                "offer a gentle high-five with its soft hand if you feel "
                "ready."
            ),
            (
                "Sit or stand near the robot and follow the breathing "
                "games on its chest screen for 10\u201320 minutes; talk "
                "to it normally and touch its hands or arms gently when "
                "it asks."
            ),
            (
                "The robot says goodbye and shows a virtual sticker on "
                "its screen; give it a high-five or wave, then walk to "
                "the door where a staff member is waiting."
            ),
        ],
        "sources": (
            '<link href="https://us.softbankrobotics.com/pepper">'
            "SoftBank Robotics</link> "
            "[13]; Logan et al. 2019 [14]; Rhee et al. 2023 [15]"
        ),
    },
    {
        "num": 6,
        "robot_full": "Humanoids",
        "robot_abbr": "Humanoids",
        "image": "6.png",
        "cancer": "pediatric bone cancer",
        "intro": (
            "This humanoid robot will walk with you and practice the route "
            "to your treatment room to help you feel prepared for your "
            "bone cancer therapy session."
        ),
        "steps": [
            (
                "Enter the activity room and approach the 1.5-meter tall "
                "robot when it waves; hold its hand if comfortable\u2014its "
                "grip is very gentle\u2014and let it kneel to your eye "
                "level over about 1 minute."
            ),
            (
                "Walk alongside the robot at your own pace (up to 1.0 m/s) "
                "for 3\u20135 minutes practicing the route, then follow "
                "its deep-breathing demonstration: inhale 1-2-3, exhale "
                "1-2-3, for 15\u201325 minutes total."
            ),
            (
                "The robot walks you back to the waiting area, waves, and "
                "says \u201cGreat job!\u201d; release its hand and walk "
                "to the door where your parent or guardian is waiting."
            ),
        ],
        "sources": (
            '<link href="https://bostondynamics.com/atlas/">Boston Dynamics</link> [16]; Darvish et al. 2024 [17]'
        ),
    },
    {
        "num": 7,
        "robot_full": "Radiotherapy Motion-Management / Tracking Robots",
        "robot_abbr": "RT Motion-Tracking Robots",
        "image": "7.png",
        "cancer": "pancreatic cancer",
        "intro": (
            "This motion-tracking system monitors your breathing in real "
            "time so the radiation beam activates only when your pancreatic "
            "tumor is precisely aligned."
        ),
        "steps": [
            (
                "Lie on the treatment couch with arms at your sides, allow "
                "a small reflective marker block to be placed on your "
                "abdomen, and breathe normally while the system calibrates "
                "in 1\u20132 minutes."
            ),
            (
                "Follow the ceiling-screen breathing guide\u2014inhale when "
                "the bar rises, exhale when it falls\u2014for 10\u201320 "
                "minutes; remain still between breaths within the 2\u20133 mm "
                "tracking tolerance."
            ),
            (
                "After the final beam, the system deactivates; hold still "
                "while the marker block is removed from your abdomen, then "
                "sit up slowly using both hands on the couch edge."
            ),
        ],
        "sources": (
            '<link href="https://www.varian.com/products/adaptive-therapy/'
            'real-time-tracking">Varian Medical</link> '
            "[18]; Bertholet et al. 2019 [19]; Keall et al. 2023 [20]"
        ),
    },
    {
        "num": 8,
        "robot_full": "Imaging Assistant Robots",
        "robot_abbr": "Imaging Robots",
        "image": "8.png",
        "cancer": "thyroid cancer",
        "intro": (
            "This robotic ultrasound system scans your neck area with "
            "consistent gentle pressure to evaluate your thyroid nodule "
            "and guide treatment planning."
        ),
        "steps": [
            (
                "Lie on the exam bed and tilt your head slightly back; "
                "place both hands at your sides and remain relaxed while "
                "ultrasound gel is applied to your neck over approximately "
                "1 minute."
            ),
            (
                "The robotic probe applies 1\u20133 N of gentle, steady "
                "pressure and scans in a grid pattern for 10\u201320 minutes; "
                "breathe normally as the robot compensates for motion "
                "automatically."
            ),
            (
                "The probe lifts away and excess gel is wiped with a warm "
                "cloth; you may sit up immediately\u2014no recovery period "
                "is needed."
            ),
        ],
        "sources": (
            "von Haxthausen et al. 2022 [21]; Jiang et al. 2023 [22]; "
            '<link href="https://www.iso.org/standard/78087.html">'
            "ISO 20417</link> [23]"
        ),
    },
    {
        "num": 9,
        "robot_full": "Steerable Needle / Needle-Steering Robots",
        "robot_abbr": "Steerable Needle Robots",
        "image": "9.png",
        "cancer": "kidney cancer",
        "intro": (
            "This needle-steering robot curves a flexible needle along a "
            "pre-planned path through tissue to reach your kidney tumor "
            "for targeted ablation treatment."
        ),
        "steps": [
            (
                "Lie on the interventional table with pillows for support, "
                "place both hands above your head, and hold still while "
                "local anesthetic is injected at the entry site (brief "
                "5-second sting)."
            ),
            (
                "The steerable needle advances along a curved path while "
                "real-time MRI confirms position every 5\u201310 seconds; "
                "remain motionless for the 30\u201360 minute procedure "
                "and breathe shallowly."
            ),
            (
                "The needle is slowly withdrawn; apply gentle pressure to "
                "the gauze on the entry site for 5\u201310 minutes, then "
                "rest on the table for 30\u201360 minutes of monitoring."
            ),
        ],
        "sources": (
            "Cowan et al. 2024 [24]; Reed et al. 2023 [25]; "
            '<link href="https://www.iso.org/standard/72424.html">'
            "ISO 7010</link> [26]"
        ),
    },
    {
        "num": 10,
        "robot_full": "Rehabilitation Exoskeletons / Robotic Gait Trainers",
        "robot_abbr": "Rehab Exoskeletons",
        "image": "10.png",
        "cancer": "bone cancer post-surgery",
        "intro": (
            "This lower-body exoskeleton supports your legs and guides "
            "your walking motion to rebuild strength and mobility after "
            "your bone cancer surgery."
        ),
        "steps": [
            (
                "Stand with your back to the exoskeleton frame, grip both "
                "parallel bars firmly, and hold steady while waist, thigh, "
                "shank, and foot straps are secured over 2\u20133 minutes."
            ),
            (
                "Follow the audible step cues and walk between the parallel "
                "bars at 0.2\u20130.5 m/s for 15\u201330 minutes; say "
                "\u201cPause\u201d to stop at any time, keeping both hands "
                "on the bars throughout."
            ),
            (
                "The exoskeleton returns you to a standing rest position; "
                "keep gripping the bars while straps are removed in "
                "reverse order (feet, shank, thigh, waist), then sit and "
                "rest for 5 minutes."
            ),
        ],
        "sources": (
            "Molteni et al. 2023 [27]; "
            '<link href="https://eksobionics.com/our-products/">'
            "Ekso Bionics</link> [28]; "
            '<link href="https://www.iso.org/standard/77326.html">'
            "ISO 15223-1</link> [12]"
        ),
    },
]


def _draw_header_footer(c, page_num, total_pages):
    """Draw consistent header bar, footer bar, and page metadata."""
    w, h = PAGE_W, PAGE_H

    # --- Top bar and author text above it ---
    c.setFont("Times-Roman", 7)
    c.drawCentredString(w / 2, h - 12 * mm, AUTHOR_LINE)
    # Top rule
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(MARGIN_L, h - 15 * mm, w - MARGIN_R, h - 15 * mm)

    # --- Bottom bar ---
    footer_y = 12 * mm
    c.setLineWidth(0.8)
    c.line(MARGIN_L, footer_y, w - MARGIN_R, footer_y)

    # Footer left: date | DOI
    c.setFont("Times-Roman", 7)
    footer_left = f"{DATE_STR}  |  {DOI}"
    c.drawString(MARGIN_L, footer_y - 9, footer_left)

    # Footer right: model | page
    footer_right = f"Claude Code Opus 4.6  |  Page {page_num} of {total_pages}"
    c.drawRightString(w - MARGIN_R, footer_y - 9, footer_right)


def _build_page(c, page_data, page_num, total_pages):
    """Build a single page of the PDF."""
    w, h = PAGE_W, PAGE_H
    y = h - 15 * mm  # below top rule

    # --- Title ---
    title_y = y - 7 * mm
    title_text = f"Patient-Robot Instructions: AI Oncology Trials - {page_data['robot_abbr']}"
    c.setFont("Times-Bold", 13)

    # Check if title is too wide and reduce font if needed
    title_width = c.stringWidth(title_text, "Times-Bold", 13)
    font_size = 13
    while title_width > CONTENT_W and font_size > 9:
        font_size -= 0.5
        title_width = c.stringWidth(title_text, "Times-Bold", font_size)

    c.setFont("Times-Bold", font_size)
    c.drawCentredString(w / 2, title_y, title_text)

    # Bar below title (same thickness as top bar)
    title_bar_y = title_y - 4 * mm
    c.setLineWidth(0.8)
    c.line(MARGIN_L, title_bar_y, w - MARGIN_R, title_bar_y)

    # --- Image ---
    img_path = os.path.join(IMAGES_DIR, page_data["image"])
    img_top_y = title_bar_y - 3 * mm
    # Image dimensions: fill width, proportional height
    img_w = CONTENT_W
    # Target image height: maximize but leave room for text below
    # We need ~60mm for text below image
    img_h = img_top_y - MARGIN_B - 62 * mm

    if os.path.exists(img_path):
        try:
            # Get actual image dimensions to maintain aspect ratio
            if PILImage:
                pil_img = PILImage.open(img_path)
                orig_w, orig_h = pil_img.size
                aspect = orig_h / orig_w
                # Fit within available space maintaining aspect ratio
                calc_h = img_w * aspect
                if calc_h > img_h:
                    # Image is taller than space, scale by height
                    img_w = img_h / aspect
                else:
                    img_h = calc_h
                pil_img.close()

            img_x = MARGIN_L + (CONTENT_W - img_w) / 2
            img_bottom_y = img_top_y - img_h
            c.drawImage(
                img_path,
                img_x,
                img_bottom_y,
                width=img_w,
                height=img_h,
                preserveAspectRatio=True,
                anchor="n",
            )
        except Exception:
            # Fallback: draw placeholder
            img_bottom_y = img_top_y - img_h
            img_x = MARGIN_L + (CONTENT_W - img_w) / 2
            c.setStrokeColor(colors.grey)
            c.setLineWidth(0.5)
            c.rect(img_x, img_bottom_y, img_w, img_h)
            c.setFont("Times-Italic", 10)
            c.drawCentredString(
                w / 2,
                img_bottom_y + img_h / 2,
                f"[Image {page_data['num']}: {page_data['robot_full']}]",
            )
    else:
        img_bottom_y = img_top_y - img_h
        img_x = MARGIN_L
        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.5)
        c.rect(img_x, img_bottom_y, img_w, img_h)
        c.setFont("Times-Italic", 10)
        c.drawCentredString(
            w / 2,
            img_bottom_y + img_h / 2,
            f"[Image {page_data['num']}: {page_data['robot_full']}]",
        )

    # --- Horizontal dashed bar below image ---
    dashed_y = img_bottom_y - 3 * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.setDash(3, 2)
    c.line(MARGIN_L, dashed_y, w - MARGIN_R, dashed_y)
    c.setDash()  # reset

    # --- Robot type full name ---
    robot_name_y = dashed_y - 5 * mm
    c.setFont("Times-Bold", 11)
    c.drawCentredString(w / 2, robot_name_y, page_data["robot_full"])

    # Bold horizontal bar under robot type (same thickness as title bar)
    robot_bar_y = robot_name_y - 3 * mm
    c.setLineWidth(0.8)
    c.line(MARGIN_L, robot_bar_y, w - MARGIN_R, robot_bar_y)

    # --- Introduction line ---
    text_y = robot_bar_y - 6 * mm
    c.setFont("Times-Roman", 9)

    # Word-wrap the intro line
    intro_text = page_data["intro"]
    text_x = MARGIN_L
    max_text_w = CONTENT_W

    # Simple word wrapping
    words = intro_text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if c.stringWidth(test_line, "Times-Roman", 9) <= max_text_w:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        c.drawString(text_x, text_y, line)
        text_y -= 11

    text_y -= 3  # small gap before numbered list

    # --- Numbered list (1-3) ---
    for i, step in enumerate(page_data["steps"], 1):
        prefix = f"{i}. "
        c.setFont("Times-Bold", 9)
        c.drawString(text_x, text_y, prefix)
        prefix_w = c.stringWidth(prefix, "Times-Bold", 9)

        c.setFont("Times-Roman", 9)
        # Word-wrap each step
        step_words = step.split()
        step_lines = []
        current_line = ""
        avail_w = max_text_w - prefix_w
        first_line = True
        for word in step_words:
            test_line = current_line + " " + word if current_line else word
            w_limit = avail_w if first_line else max_text_w - 12
            if c.stringWidth(test_line, "Times-Roman", 9) <= w_limit:
                current_line = test_line
            else:
                step_lines.append((current_line, first_line))
                current_line = word
                first_line = False
        if current_line:
            step_lines.append((current_line, first_line))

        for line_text, is_first in step_lines:
            if is_first:
                c.drawString(text_x + prefix_w, text_y, line_text)
            else:
                c.drawString(text_x + 12, text_y, line_text)
            text_y -= 11

        text_y -= 2  # gap between items

    # --- Bottom section: sources line ---
    # Position sources just above the footer bar
    footer_y = 12 * mm
    source_y = footer_y + 4 * mm

    c.setLineWidth(0.4)
    c.line(MARGIN_L, source_y + 5, w - MARGIN_R, source_y + 5)

    c.setFont("Times-Roman", 6.5)
    # Strip HTML-like link tags for plain text rendering
    sources_plain = page_data["sources"]
    sources_clean = re.sub(r"<link[^>]*>", "", sources_plain)
    sources_clean = sources_clean.replace("</link>", "")
    sources_text = f"Sources: {sources_clean}"

    # Truncate if too long
    max_source_w = CONTENT_W * 0.65
    if c.stringWidth(sources_text, "Times-Roman", 6.5) > max_source_w:
        while c.stringWidth(sources_text + "...", "Times-Roman", 6.5) > max_source_w and len(sources_text) > 20:
            sources_text = sources_text[:-1]
        sources_text += "..."

    c.drawString(MARGIN_L, source_y - 3, sources_text)

    # "For Demonstration Purposes Only" to the right of sources
    demo_text = "For Demonstration Purposes Only"
    c.setFont("Times-Italic", 6.5)
    c.drawRightString(w - MARGIN_R, source_y - 3, demo_text)


def generate_pdf(output_path, quality="full"):
    """Generate the 10-page PDF.

    Args:
        output_path: Path for the output PDF file.
        quality: 'full' for maximum quality, '10mb' or '5mb' for compressed.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle("Patient-Robot Instructions: Physical AI Oncology Trials")
    c.setAuthor("Kevin Kawchak, CEO ChemicalQDevice")
    c.setSubject("Patient-facing instructional sheets for 10 robot types")

    total_pages = len(PAGES)

    for page_data in PAGES:
        _draw_header_footer(c, page_data["num"], total_pages)
        _build_page(c, page_data, page_data["num"], total_pages)
        c.showPage()

    c.save()
    print(f"Generated: {output_path} ({os.path.getsize(output_path):,} bytes)")


def compress_pdf(input_path, output_path, target_mb):
    """Create a compressed version of the PDF targeting a specific file size.

    Uses Pillow to reduce image quality in a re-generated PDF.
    """
    global IMAGES_DIR
    target_bytes = target_mb * 1024 * 1024

    if not PILImage:
        shutil.copy2(input_path, output_path)
        return

    src_size = os.path.getsize(input_path)
    if src_size <= target_bytes:
        shutil.copy2(input_path, output_path)
        print(f"Compressed: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return

    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="pdf_compress_")
    orig_images_dir = IMAGES_DIR
    try:
        ratio = target_bytes / src_size
        scale = max(0.3, min(1.0, ratio**0.5))
        use_jpg = target_mb <= 5

        for page_data in PAGES:
            src_img = os.path.join(orig_images_dir, page_data["image"])
            if os.path.exists(src_img):
                img = PILImage.open(src_img)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                if use_jpg:
                    dst_name = page_data["image"].replace(".png", ".jpg")
                    dst_img = os.path.join(temp_dir, dst_name)
                    if img.mode == "RGBA":
                        bg = PILImage.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    img.save(dst_img, "JPEG", quality=65, optimize=True)
                else:
                    dst_img = os.path.join(temp_dir, page_data["image"])
                    img.save(dst_img, "PNG", optimize=True)
                img.close()

        IMAGES_DIR = temp_dir
        if use_jpg:
            for page_data in PAGES:
                page_data["image"] = page_data["image"].replace(".png", ".jpg")

        generate_pdf(output_path)

    finally:
        IMAGES_DIR = orig_images_dir
        for page_data in PAGES:
            page_data["image"] = page_data["image"].replace(".jpg", ".png")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Compressed: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    """Generate all three PDF versions."""
    os.makedirs(PAPER_DIR, exist_ok=True)

    # Full-size PDF
    full_path = os.path.join(PAPER_DIR, "Patient-Robot Instructions: Physical AI Oncology Trials.pdf")
    generate_pdf(full_path, quality="full")

    # 10 MB version
    pdf_10mb = os.path.join(
        PAPER_DIR,
        "Patient-Robot Instructions: Physical AI Oncology Trials (10MB).pdf",
    )
    compress_pdf(full_path, pdf_10mb, 10)

    # 5 MB version
    pdf_5mb = os.path.join(
        PAPER_DIR,
        "Patient-Robot Instructions: Physical AI Oncology Trials (5MB).pdf",
    )
    compress_pdf(full_path, pdf_5mb, 5)


if __name__ == "__main__":
    main()
