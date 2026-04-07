# RTX 4090 GDDR6X / 24GB - OpenClaw 168-Hour Real-Time Simulation

Instructions for running the 168-hour autonomous sponsor simulation in real time on an NVIDIA RTX 4090 (24GB GDDR6X) workstation with OpenClaw integration.

## Hardware Requirements

- GPU: NVIDIA RTX 4090 (24GB GDDR6X VRAM)
- CPU: AMD Ryzen 9 7950X or Intel Core i9-13900K (or equivalent)
- RAM: 64GB DDR5 minimum
- Storage: 1TB NVMe SSD (for simulation data and logs)
- Network: Stable internet connection for GitHub push operations

---

## 1. Linux Ubuntu (24.04 LTS / 24.10)

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (535+ recommended for RTX 4090)
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA Toolkit 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# Verify GPU
nvidia-smi
```

### Python Environment

```bash
# Install Python 3.10+ (included in Ubuntu 24.04+)
sudo apt install -y python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### OpenClaw Setup

```bash
# Install OpenClaw (Isaac Sim based)
pip install openclaw

# Verify OpenClaw can access the RTX 4090
python3 -c "import openclaw; print(openclaw.get_gpu_info())"
```

### Run 168-Hour Simulation

```bash
cd sponsor/final_paper/168_hours

# Option A: Run all 168 hours sequentially (fast mode, no real-time delay)
python3 run_168h_simulation.py

# Option B: Run with real-time pacing (1 hour of computation per real hour)
# Use cron or a systemd timer to execute each hour:
for hour in $(seq 0 167); do
    python3 _gen_hourly.py $hour
    sleep 3600  # Wait 1 real hour before next iteration
done

# Option C: Use cron for automated hourly execution
# Add to crontab (crontab -e):
# 0 * * * * cd /path/to/physical-ai-oncology-trials/sponsor/final_paper/168_hours && python3 _gen_hourly.py $(( ($(date +\%s) - $(date -d '2026-03-23' +\%s)) / 3600 ))
```

---

## 2. macOS (Sequoia 15.x / Sonoma 14.x)

Note: The RTX 4090 is not natively supported on macOS. These instructions assume an eGPU enclosure with Thunderbolt 4 or a Hackintosh configuration. For native macOS GPU support, see the Mac Mini M4 Pro instructions instead.

### Prerequisites

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.12 git

# Note: NVIDIA CUDA is not supported on macOS.
# The simulation runs in CPU-only mode on macOS with RTX 4090 via eGPU.
```

### Python Environment

```bash
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Run Simulation

```bash
cd sponsor/final_paper/168_hours
python3 run_168h_simulation.py

# For real-time pacing with launchd:
# Create a plist file in ~/Library/LaunchAgents/ for hourly execution
```

---

## 3. Windows (Windows 11 24H2 / Windows Server 2025)

### Prerequisites

```powershell
# Install NVIDIA drivers from https://www.nvidia.com/Download/index.aspx
# Select: GeForce RTX 4090, Windows 11 64-bit

# Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
# Select: Windows, x86_64, exe (network)

# Verify GPU in PowerShell
nvidia-smi
```

### Python Environment

```powershell
# Install Python 3.12 from https://www.python.org/downloads/
# Ensure "Add Python to PATH" is checked during installation

# Install Git from https://git-scm.com/download/win

# Clone repository
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### OpenClaw Setup

```powershell
pip install openclaw
python -c "import openclaw; print(openclaw.get_gpu_info())"
```

### Run Simulation

```powershell
cd sponsor\final_paper\168_hours

# Run all 168 hours
python run_168h_simulation.py

# For real-time pacing, use Windows Task Scheduler:
# Create a task that runs hourly:
# Action: python.exe _gen_hourly.py %HOUR%
# Trigger: Repeat every 1 hour for 168 hours
```

---

## Performance Notes

- The RTX 4090 provides 82.6 TFLOPS FP32, enabling real-time robot simulation during sponsor operations
- Full 168-hour standalone simulation completes in under 5 seconds (no GPU required for decision generation)
- GPU acceleration is used for OpenClaw robot physics simulation and digital twin rendering
- VRAM usage: approximately 8-12GB during peak robot simulation (14 active robot instances)
