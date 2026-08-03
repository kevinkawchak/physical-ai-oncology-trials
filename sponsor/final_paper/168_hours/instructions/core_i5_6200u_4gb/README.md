# Core i5-6200U 4GB - 168-Hour Real-Time Simulation (Windows 10 Pro)

Instructions for running the 168-hour autonomous sponsor simulation in real time on an Intel Core i5-6200U laptop (4GB RAM) with Windows 10 Pro. This guide provides exact step-by-step instructions for basic and advanced autonomous execution methods.

## Hardware Specifications

- **Device Name:** DESKTOP-NM1S06E
- **Processor:** Intel Core i5-6200U CPU @ 2.30GHz (2 cores, 4 threads, 2.40 GHz turbo)
- **Installed RAM:** 4.00 GB (3.89 GB usable)
- **System Type:** 64-bit operating system, x64-based processor
- **OS:** Windows 10 Pro, Version 21H2, Build 19044.2965
- **Free Storage:** 176 GB

## Hardware Limitations and Potential Issues

Before starting, be aware of the following hardware constraints that may affect the 168-hour run:

1. **4GB RAM is extremely limited.** Windows 10 itself uses 1.5-2.5 GB at idle. Python with virtual environment overhead leaves only 1-2 GB for the simulation. Close ALL unnecessary applications (browsers, Office, OneDrive, Cortana, etc.) before and during the run.
2. **Thermal throttling.** The i5-6200U is a 15W mobile processor. Running continuously for 168 hours (7 days) will generate sustained heat. Ensure the laptop is on a hard, flat surface with good airflow. Consider elevating the back of the laptop or using a cooling pad. Do NOT place the laptop on a bed, couch, or other soft surface that blocks vents.
3. **Power management.** The laptop MUST remain plugged into AC power for the entire 168-hour run. A power interruption will stop the simulation. Use a surge protector or UPS (uninterruptible power supply) if available.
4. **Sleep and hibernation.** Windows will attempt to sleep or hibernate the laptop. This MUST be disabled (instructions below). Even with settings changed, Windows Update may force a restart - disable automatic updates before starting.
5. **Disk space.** The simulation generates approximately 50 MB of output files total. With 176 GB free this is not a concern, but ensure no other large downloads or updates consume disk space during the run.
6. **Screen and display.** The display can be turned off to save power, but the system must not sleep. The simulation runs headlessly (no display required).
7. **Windows Update restarts.** Windows 10 21H2 may force-restart for updates. This is the single biggest risk to completing a 168-hour run. Follow the Windows Update disable steps carefully.
8. **Battery wear.** Keeping the laptop plugged in for 7 days continuously is safe for modern lithium-ion batteries. The battery management circuit handles this. If the laptop supports it, set the battery charge limit to 80% in BIOS/UEFI settings to reduce wear.
9. **Antivirus interference.** Windows Defender real-time scanning can slow Python execution and cause CPU spikes. Consider adding the repository folder to the exclusion list.
10. **Pagefile usage.** With only 4GB RAM, Windows will use the pagefile (virtual memory on disk) frequently. This is normal but will slow execution. The simulation scripts are lightweight (standard library only) and should not exceed available memory if other applications are closed.

---

## Prerequisites (Both Methods)

These steps must be completed before starting either Method A or Method B.

### Step 1: Install Python

1. Open your web browser (Edge or Chrome)
2. Go to: `https://www.python.org/downloads/`
3. Click the yellow **"Download Python 3.12.x"** button (or the latest 3.12 release)
4. When the installer downloads, click on it to run it
5. **IMPORTANT:** On the first installer screen, check the box that says **"Add python.exe to PATH"** at the bottom
6. Click **"Install Now"**
7. Wait for installation to complete, then click **"Close"**
8. Verify the installation:
   - Press `Win + R` on your keyboard
   - Type `cmd` and press Enter
   - In the black Command Prompt window, type: `python --version`
   - You should see something like `Python 3.12.x`
   - If you see `Python was not found` or an error, restart your computer and try again
   - If it still does not work, uninstall Python from Settings > Apps, then reinstall making sure to check "Add python.exe to PATH"

### Step 2: Install Git

1. Open your web browser
2. Go to: `https://git-scm.com/download/win`
3. The download should start automatically for the 64-bit version. If not, click **"64-bit Git for Windows Setup"**
4. Run the downloaded installer
5. Click **Next** through all screens, accepting all defaults. Do not change any settings
6. Click **Install**, then **Finish**
7. Verify the installation:
   - Open a NEW Command Prompt window (press `Win + R`, type `cmd`, press Enter)
   - Type: `git --version`
   - You should see something like `git version 2.x.x`

### Step 3: Clone the Repository

1. Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
2. Navigate to a folder where you want the project. Type each line and press Enter:

```cmd
cd %USERPROFILE%\Desktop
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials
```

3. If the clone fails with a network error, try again. If it still fails, check your internet connection.

### Step 4: Create a Python Virtual Environment

In the same Command Prompt window (you should be inside the `physical-ai-oncology-trials` folder):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

After activation, you should see `(.venv)` at the beginning of your command line. If you see an error about "running scripts is disabled", open PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then close PowerShell and try the activation again in Command Prompt.

### Step 5: Install Dependencies

```cmd
pip install -r requirements.txt
```

If this fails due to packages requiring compilation (such as torch or scipy), that is OK. The 168-hour simulation uses only the Python standard library and does not require any external packages. You can skip this step if it fails.

### Step 6: Prevent Sleep, Hibernation, and Automatic Restarts

This is critical for a 168-hour uninterrupted run.

**Disable Sleep and Hibernation:**

1. Click the **Start** button (Windows icon, bottom-left)
2. Click the **gear icon** (Settings)
3. Click **System**
4. Click **Power & sleep** in the left sidebar
5. Under **Screen**, set both dropdowns to **Never**
6. Under **Sleep**, set both dropdowns to **Never**
7. Close Settings

**Disable Hibernation (Command Prompt as Administrator):**

1. Click the **Start** button
2. Type `cmd`
3. Right-click on **Command Prompt** and select **Run as administrator**
4. Click **Yes** on the User Account Control prompt
5. Type the following and press Enter:

```cmd
powercfg -h off
```

6. Close the Administrator Command Prompt

**Disable Windows Automatic Updates (Temporary, for 168 hours):**

1. Press `Win + R`
2. Type `services.msc` and press Enter
3. Scroll down to find **Windows Update**
4. Double-click on it
5. In the **Startup type** dropdown, select **Disabled**
6. Click the **Stop** button if the service status shows "Running"
7. Click **OK**
8. **IMPORTANT:** After the 168-hour simulation completes, re-enable this service by repeating these steps and setting Startup type back to **Automatic**

**Disable Windows Defender Real-Time Scanning for the Repository Folder (Optional but Recommended):**

1. Click the **Start** button
2. Type `Windows Security` and open it
3. Click **Virus & threat protection**
4. Under **Virus & threat protection settings**, click **Manage settings**
5. Scroll down to **Exclusions** and click **Add or remove exclusions**
6. Click **Add an exclusion** > **Folder**
7. Navigate to `C:\Users\YourUsername\Desktop\physical-ai-oncology-trials` and select it
8. **IMPORTANT:** Remove this exclusion after the simulation completes

---

## Method A: Task Scheduler (Recommended - Hourly Execution)

This method uses Windows Task Scheduler to run one simulation hour every real hour for 168 hours. This is the most reliable method for a 168-hour autonomous run because each hour runs independently, so a single script failure does not stop the entire simulation.

### Step A1: Create the Hourly Runner Script

1. Open **Notepad** (press `Win + R`, type `notepad`, press Enter)
2. Copy and paste the following EXACTLY into Notepad:

```python
"""Hourly runner for 168-hour simulation via Task Scheduler."""

import subprocess
import sys
import os
import json
from datetime import datetime


def get_current_hour():
    """Read the current hour counter from the state file."""
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hour_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            data = json.load(f)
            return data.get("current_hour", 0)
    return 0


def save_current_hour(hour):
    """Save the current hour counter to the state file."""
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hour_state.json")
    with open(state_file, "w") as f:
        json.dump({"current_hour": hour, "timestamp": datetime.now().isoformat()}, f, indent=2)


def main():
    """Execute the current hour and increment the counter."""
    current_hour = get_current_hour()
    if current_hour >= 168:
        print(f"Simulation complete. All 168 hours have been executed.")
        return

    sim_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(sim_dir, "scheduler_log.txt")

    with open(log_file, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] Starting hour {current_hour:03d}\n")

    day = (current_hour // 24) + 1
    hour_in_day = current_hour % 24
    script_name = f"sponsor_hour_{current_hour:03d}.py"
    script_path = os.path.join(sim_dir, f"day_{day:02d}", "hourly", script_name)

    if os.path.exists(script_path):
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.join(sim_dir, f"day_{day:02d}", "hourly"),
            timeout=3000,
        )
        with open(log_file, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] Hour {current_hour:03d} exit code: {result.returncode}\n")
            if result.stderr:
                log.write(f"  STDERR: {result.stderr[:500]}\n")
    else:
        with open(log_file, "a") as log:
            log.write(f"[{datetime.now().isoformat()}] WARNING: Script not found: {script_path}\n")

    save_current_hour(current_hour + 1)

    with open(log_file, "a") as log:
        remaining = 167 - current_hour
        log.write(f"[{datetime.now().isoformat()}] Hour {current_hour:03d} done. {remaining} hours remaining.\n")


if __name__ == "__main__":
    main()
```

3. In Notepad, click **File** > **Save As**
4. In the **"Save as type"** dropdown at the bottom, change it from "Text Documents (*.txt)" to **"All Files (*.*)"**
5. Navigate to: `C:\Users\YourUsername\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours`
6. In the **"File name"** field, type: `run_hourly_task.py`
7. Click **Save**

### Step A2: Test the Script Manually

1. Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
2. Type the following (adjust your username):

```cmd
cd %USERPROFILE%\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours
python run_hourly_task.py
```

3. You should see no errors. Check that `hour_state.json` was created in the 168_hours folder and contains `"current_hour": 1`.
4. Check that `scheduler_log.txt` was created and contains log entries.
5. If you want to reset the counter to start from hour 0, delete `hour_state.json`.

### Step A3: Create the Scheduled Task

1. Press the **Start** button
2. Type `Task Scheduler` and open it
3. In the right panel, click **Create Task** (not "Create Basic Task")
4. **General tab:**
   - Name: `SponsorSim168h`
   - Description: `168-hour autonomous sponsor simulation`
   - Check **"Run whether user is logged on or not"**
   - Check **"Run with highest privileges"**
5. **Triggers tab:**
   - Click **New**
   - Set **"Begin the task"** to **"On a schedule"**
   - Select **"One time"**
   - Set the **start date and time** to when you want the simulation to begin (e.g., now or a few minutes from now)
   - Check **"Repeat task every"** and set to **"1 hour"**
   - Set **"for a duration of"** to **"8 days"** (this gives a buffer beyond 168 hours / 7 days)
   - Check **"Enabled"**
   - Click **OK**
6. **Actions tab:**
   - Click **New**
   - Action: **"Start a program"**
   - Program/script: `python.exe`
   - Add arguments: `run_hourly_task.py`
   - Start in: `C:\Users\YourUsername\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours`
   - Replace `YourUsername` with your actual Windows username
   - Click **OK**
7. **Conditions tab:**
   - UNCHECK **"Start the task only if the computer is on AC power"**
   - UNCHECK **"Stop if the computer switches to battery power"**
   - UNCHECK all other conditions
8. **Settings tab:**
   - Check **"Allow task to be run on demand"**
   - Check **"If the task fails, restart every"** and set to **"1 minute"**, attempt to restart up to **"3"** times
   - Set **"Stop the task if it runs longer than"** to **"1 hour"**
   - Set **"If the task is already running"** to **"Do not start a new instance"**
   - Click **OK**
9. You may be prompted for your Windows password. Enter it and click **OK**.

### Step A4: Verify the Task is Running

1. In Task Scheduler, find **SponsorSim168h** in the Task Scheduler Library
2. Right-click it and select **Run** to test it immediately
3. Check the **Last Run Result** column. It should show **(0x0)** for success
4. Open the file `scheduler_log.txt` in the 168_hours folder to verify execution
5. Check that `hour_state.json` shows the hour counter incrementing

**Troubleshooting Task Scheduler:**

- If the Last Run Result shows **(0x1)** or another error code, the Python path may not be found. In the Actions tab, change "Program/script" from `python.exe` to the full path: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe` (adjust for your actual Python installation path). To find your Python path, open Command Prompt and type: `where python`
- If the task does not run at all, ensure "Run whether user is logged on or not" is checked and you entered your password correctly
- If you see "Access Denied" errors in the log, re-create the task with "Run with highest privileges" checked

### Step A5: Monitor Progress

During the 168-hour run, you can check progress at any time:

1. Open the file `scheduler_log.txt` in the 168_hours folder with Notepad to see all executed hours
2. Open `hour_state.json` to see the current hour number
3. Check the output folders (`day_01/hourly/output/`, `day_02/hourly/output/`, etc.) for JSON output files

### Step A6: After Completion

After 168 hours (7 days), the simulation will complete. The `run_hourly_task.py` script will stop processing once it reaches hour 168.

1. Open Task Scheduler
2. Find **SponsorSim168h**
3. Right-click and select **Delete** to remove the scheduled task
4. Re-enable Windows Update (see Step 6 above)
5. Remove the Windows Defender exclusion if you added one

Output files will be located in:
- `sponsor/final_paper/168_hours/day_01/hourly/output/` through `day_07/hourly/output/` (168 JSON files)
- `sponsor/final_paper/168_hours/day_01/output/` through `day_07/output/` (7 daily summary JSON files)
- `sponsor/final_paper/168_hours/scheduler_log.txt` (execution log)

---

## Method B: Continuous Python Loop (Alternative)

This method runs a single Python script that loops through all 168 hours with a 1-hour real-time delay between each. This is simpler to set up but less resilient: if the script crashes, all remaining hours are lost unless you manually restart.

### Step B1: Create the Continuous Runner Script

1. Open **Notepad** (press `Win + R`, type `notepad`, press Enter)
2. Copy and paste the following EXACTLY into Notepad:

```python
"""Continuous 168-hour real-time simulation runner."""

import subprocess
import sys
import os
import time
import json
from datetime import datetime


def main():
    """Run all 168 hours with 1-hour real-time delays."""
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(sim_dir, "continuous_log.txt")
    state_file = os.path.join(sim_dir, "continuous_state.json")

    # Resume from last completed hour if restarting after a crash
    start_hour = 0
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            data = json.load(f)
            start_hour = data.get("next_hour", 0)
        print(f"Resuming from hour {start_hour:03d}")

    with open(log_file, "a") as log:
        log.write(f"\n[{datetime.now().isoformat()}] === Simulation started (hour {start_hour:03d}) ===\n")

    for hour in range(start_hour, 168):
        day = (hour // 24) + 1
        script_name = f"sponsor_hour_{hour:03d}.py"
        script_path = os.path.join(sim_dir, f"day_{day:02d}", "hourly", script_name)

        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] Running hour {hour:03d} of 167 (Day {day}, {167 - hour} remaining)")

        with open(log_file, "a") as log:
            log.write(f"[{timestamp}] Starting hour {hour:03d}\n")

        if os.path.exists(script_path):
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=os.path.join(sim_dir, f"day_{day:02d}", "hourly"),
                    timeout=3000,
                )
                with open(log_file, "a") as log:
                    log.write(f"[{datetime.now().isoformat()}] Hour {hour:03d} exit code: {result.returncode}\n")
                    if result.stderr:
                        log.write(f"  STDERR: {result.stderr[:500]}\n")
            except subprocess.TimeoutExpired:
                with open(log_file, "a") as log:
                    log.write(f"[{datetime.now().isoformat()}] Hour {hour:03d} TIMED OUT\n")
            except Exception as e:
                with open(log_file, "a") as log:
                    log.write(f"[{datetime.now().isoformat()}] Hour {hour:03d} ERROR: {e}\n")
        else:
            with open(log_file, "a") as log:
                log.write(f"[{datetime.now().isoformat()}] WARNING: Script not found: {script_path}\n")

        # Save state after each hour for crash recovery
        with open(state_file, "w") as f:
            json.dump({"next_hour": hour + 1, "timestamp": datetime.now().isoformat()}, f, indent=2)

        # Wait 1 real hour before next iteration (skip wait on last hour)
        if hour < 167:
            print(f"  Waiting 3600 seconds (1 hour) until next execution...")
            time.sleep(3600)

    with open(log_file, "a") as log:
        log.write(f"[{datetime.now().isoformat()}] === Simulation complete (all 168 hours) ===\n")

    print("Simulation complete. All 168 hours executed.")


if __name__ == "__main__":
    main()
```

3. In Notepad, click **File** > **Save As**
4. Change **"Save as type"** to **"All Files (*.*)"**
5. Navigate to: `C:\Users\YourUsername\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours`
6. In the **"File name"** field, type: `run_168h_realtime.py`
7. Click **Save**

### Step B2: Start the Continuous Simulation

1. Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
2. Type the following:

```cmd
cd %USERPROFILE%\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours
python run_168h_realtime.py
```

3. The script will begin running. You will see output like:

```
[2026-04-07T10:00:00] Running hour 000 of 167 (Day 1, 167 remaining)
  Waiting 3600 seconds (1 hour) until next execution...
```

4. **Do not close the Command Prompt window.** Minimizing it is OK.
5. If you accidentally close it or the system restarts, re-run the same command. The script will automatically resume from the last completed hour using `continuous_state.json`.

### Step B3: Keep the Command Prompt Window Open

To prevent accidental closure:

1. Do NOT close the Command Prompt window for 7 days
2. If you need to use Command Prompt for something else, open a NEW window (press `Win + R`, type `cmd`, press Enter) rather than using the one running the simulation
3. Consider using `start /min` to start it minimized:

```cmd
cd %USERPROFILE%\Desktop\physical-ai-oncology-trials\sponsor\final_paper\168_hours
start /min python run_168h_realtime.py
```

**Troubleshooting Continuous Runner:**

- If the script crashes or the system restarts, simply re-run `python run_168h_realtime.py` and it will resume from where it left off
- Check `continuous_log.txt` for the execution history
- Check `continuous_state.json` to see the next hour to be executed
- If you want to restart from the beginning, delete `continuous_state.json`

### Step B4: Monitor Progress

During the 168-hour run:

1. The Command Prompt window shows the current hour being executed
2. Open `continuous_log.txt` with Notepad for the full execution history
3. Open `continuous_state.json` to see the next scheduled hour
4. Check output folders as described in Method A, Step A5

Output files will be located in the same directories as Method A (see Step A6 above).

---

## Comparing Methods

| Feature | Method A: Task Scheduler | Method B: Continuous Loop |
|---------|--------------------------|---------------------------|
| Setup complexity | More steps, but one-time | Simple, just run a script |
| Resilience | High - survives crashes, logoffs | Medium - survives crashes via state file, but must manually restart |
| Survives reboot | Yes (task auto-resumes) | No (must manually restart the script) |
| Resource usage | Low (Python starts/stops each hour) | Low (Python sleeps between hours) |
| Memory pressure on 4GB system | Better (Python releases memory between runs) | Slightly worse (Python process stays resident) |
| Monitoring | Task Scheduler UI + log file | Command Prompt window + log file |
| Recommended for 4GB RAM | Yes | Yes, if system is stable |

**Recommendation for this hardware:** Method A (Task Scheduler) is preferred because it releases Python from memory between hourly runs, reducing memory pressure on the 4GB system. It also survives system reboots automatically.

---

## Output File Locations

After the simulation completes (either method), the following output files will be present on the laptop:

```
physical-ai-oncology-trials/
  sponsor/final_paper/168_hours/
    scheduler_log.txt                    # Method A execution log
    hour_state.json                      # Method A state tracker
    continuous_log.txt                   # Method B execution log
    continuous_state.json                # Method B state tracker
    day_01/
      hourly/
        output/
          sponsor_hour_000_output.json   # Hour 0 output
          sponsor_hour_001_output.json   # Hour 1 output
          ...
          sponsor_hour_023_output.json   # Hour 23 output
      output/
        day_01_summary.json              # Day 1 cumulative summary
    day_02/
      hourly/output/                     # Hours 24-47 output JSON files
      output/day_02_summary.json         # Day 2 cumulative summary
    day_03/
      hourly/output/                     # Hours 48-71 output JSON files
      output/day_03_summary.json         # Day 3 cumulative summary
    day_04/
      hourly/output/                     # Hours 72-95 output JSON files
      output/day_04_summary.json         # Day 4 cumulative summary
    day_05/
      hourly/output/                     # Hours 96-119 output JSON files
      output/day_05_summary.json         # Day 5 cumulative summary
    day_06/
      hourly/output/                     # Hours 120-143 output JSON files
      output/day_06_summary.json         # Day 6 cumulative summary
    day_07/
      hourly/output/                     # Hours 144-167 output JSON files
      output/day_07_summary.json         # Day 7 cumulative summary
```

## Performance Notes

- The i5-6200U provides 2 cores / 4 threads at 2.30 GHz base, 2.80 GHz turbo. Each hourly script completes in under 1 second; the real-time pacing is entirely from the 1-hour wait between executions.
- With 4GB RAM, memory is the primary constraint. The simulation scripts use Python standard library only and typically consume 20-40 MB per execution. Task Scheduler (Method A) is preferred as it releases this memory between runs.
- The 176 GB of free storage is more than sufficient. Total output (168 JSON files + 7 summaries + log files) is approximately 50 MB.
- No GPU is required. The simulation generates deterministic sponsor decisions using CPU-only computation.
- The laptop should remain in a well-ventilated area for the full 7-day run. While individual hourly scripts run briefly, the system must remain powered on and awake continuously.
