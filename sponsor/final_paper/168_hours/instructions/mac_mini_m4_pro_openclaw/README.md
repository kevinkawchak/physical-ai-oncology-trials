# Mac Mini M4 Pro 64GB - OpenClaw 168-Hour Real-Time Simulation

Instructions for running the 168-hour autonomous sponsor simulation in real time on an Apple Mac Mini M4 Pro (64GB unified memory) with OpenClaw integration.

## Hardware Requirements

- SoC: Apple M4 Pro (14-core CPU, 20-core GPU, 16-core Neural Engine)
- RAM: 64GB unified memory
- Storage: 1TB SSD minimum
- Network: Stable internet connection for GitHub push operations

---

## 1. Linux Ubuntu (24.04 LTS / 24.10 - ARM64)

### Prerequisites

```bash
# Install Ubuntu for Apple Silicon using Asahi Linux or UTM virtualization
# Asahi Linux: https://asahilinux.org/
# UTM: https://mac.getutm.app/

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv git build-essential
```

### Python Environment

```bash
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### OpenClaw Setup

```bash
# Install OpenClaw (ARM64 compatible build)
pip install openclaw

# Verify OpenClaw detects the M4 Pro GPU via Metal/MPS
python3 -c "import openclaw; print(openclaw.get_device_info())"
```

### Run 168-Hour Simulation

```bash
cd sponsor/final_paper/168_hours

# Fast mode: run all 168 hours sequentially
python3 run_168h_simulation.py

# Real-time pacing with systemd timer:
# Create /etc/systemd/system/sponsor-sim.service and sponsor-sim.timer
# Timer triggers hourly for 168 hours

# Alternative: cron-based execution
# 0 * * * * cd /path/to/repo/sponsor/final_paper/168_hours && python3 _gen_hourly.py $(( ($(date +\%s) - $(date -d '2026-03-23' +\%s)) / 3600 ))
```

---

## 2. macOS (Sequoia 15.x / Sonoma 14.x)

### Prerequisites

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.12 git
```

### Python Environment

```bash
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### OpenClaw Setup

```bash
# Install OpenClaw with Metal backend
pip install openclaw

# Verify Metal GPU acceleration on M4 Pro
python3 -c "import openclaw; print(openclaw.get_device_info())"

# The M4 Pro uses Metal Performance Shaders (MPS) for GPU acceleration
# 64GB unified memory provides direct GPU access without data transfers
```

### Run 168-Hour Simulation

```bash
cd sponsor/final_paper/168_hours

# Fast mode
python3 run_168h_simulation.py

# Real-time pacing with launchd:
# Create ~/Library/LaunchAgents/com.sponsor.sim.plist
cat > ~/Library/LaunchAgents/com.sponsor.sim.plist << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sponsor.sim</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>run_168h_simulation.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/repo/sponsor/final_paper/168_hours</string>
    <key>StartInterval</key>
    <integer>3600</integer>
</dict>
</plist>
PLIST

# Load the launch agent
launchctl load ~/Library/LaunchAgents/com.sponsor.sim.plist
```

---

## 3. Windows (Windows 11 24H2 - ARM64)

### Prerequisites

```powershell
# Windows 11 ARM64 runs natively on Mac Mini M4 Pro via:
# - Parallels Desktop (recommended)
# - VMware Fusion
# - Boot Camp is NOT available for Apple Silicon

# Install Python 3.12 (ARM64) from https://www.python.org/downloads/
# Install Git from https://git-scm.com/download/win
```

### Python Environment

```powershell
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### OpenClaw Setup

```powershell
# OpenClaw on Windows ARM64 uses DirectX/DirectML for GPU acceleration
pip install openclaw

python -c "import openclaw; print(openclaw.get_device_info())"
```

### Run Simulation

```powershell
cd sponsor\final_paper\168_hours

# Fast mode
python run_168h_simulation.py

# Real-time pacing with Task Scheduler:
# Create hourly task via PowerShell:
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "run_168h_simulation.py" -WorkingDirectory "C:\path\to\repo\sponsor\final_paper\168_hours"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 168)
Register-ScheduledTask -TaskName "SponsorSim168h" -Action $action -Trigger $trigger
```

---

## Performance Notes

- The M4 Pro with 64GB unified memory provides 273 GB/s memory bandwidth
- GPU and CPU share the same memory pool, eliminating data transfer overhead
- The 20-core GPU delivers approximately 19 TFLOPS FP32 for physics simulation
- Neural Engine (16-core) can be leveraged for ML inference in real-time
- Full 168-hour standalone simulation completes in under 3 seconds
- Robot physics simulation via OpenClaw uses Metal backend with MPS acceleration
- Unified memory architecture is particularly efficient for large model contexts
