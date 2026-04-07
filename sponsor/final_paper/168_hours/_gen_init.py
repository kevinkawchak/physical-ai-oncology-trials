"""Generate __init__.py files for each day's hourly directory."""

from __future__ import annotations

from pathlib import Path

from _config import DAY_THEMES

BASE = Path(__file__).resolve().parent


def generate_day_init(day):
    """Create __init__.py for a day's hourly directory."""
    theme = DAY_THEMES[day]
    start_hour = (day - 1) * 24
    end_hour = start_hour + 23
    content = f'''"""Day {day} hourly sponsor activity scripts - {theme}.

Contains sponsor_hour_{start_hour:03d}.py through sponsor_hour_{end_hour:03d}.py,
each generating 12 sponsor decisions at 5-minute intervals.
"""
'''
    day_dir = BASE / f"day_{day:02d}" / "hourly"
    day_dir.mkdir(parents=True, exist_ok=True)
    init_path = day_dir / "__init__.py"
    init_path.write_text(content)
    return init_path


if __name__ == "__main__":
    for d in range(1, 8):
        p = generate_day_init(d)
        print(f"  {p}")
