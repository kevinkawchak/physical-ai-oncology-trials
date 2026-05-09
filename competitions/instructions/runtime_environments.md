# Runtime Environments

This file fixes the leading MacOS, Windows, and Linux execution recipes for the future simulation. The future session must include the recipes verbatim in `competitions/glioblastoma-1hr-trial/README.md` and must validate that each recipe runs to completion before considering Commit 1 complete.

## Required Local Toolchain

| Tool | Minimum version | Purpose |
|------|-----------------|---------|
| Python | 3.10 | Orchestration, ingest, mapping, metrics, LLM agent |
| pip | 23.0 | Package install |
| Rust | 1.75 | High-throughput iteration runner |
| C++ compiler | g++ 11 or clang 14 | Real-time control loop |
| Docker | 24.0 | Optional multi-service stack |
| Git | 2.40 | Version control and competition snapshots |
| DuckDB CLI | 0.9 | Optional analytical queries |

GPU is optional. The future simulation runs on CPU because the dynamics model is closed-form rigid-body kinematics for a 6-DOF arm. The future Commit 5 LLM agent uses Anthropic API calls or a local Ollama deployment of an open-weight model.

## Linux Recipe (Ubuntu 22.04 LTS)

```
# 1. System packages
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git curl build-essential

# 2. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 3. Clone and enter
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials/competitions/glioblastoma-1hr-trial

# 4. Python venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# 5. Build the Rust runner
cargo build --release --manifest-path src/simulation/Cargo.toml

# 6. Build the C++ control loop
g++ -std=c++20 -O2 -o build/robot_loop src/control/robot_loop.cpp

# 7. Run a single iteration smoke test
python -m src.simulation.iterate --seed 20260509 --iterations 1 --out data/iterations
```

## MacOS Recipe (Apple Silicon, macOS 14 Sonoma)

```
# 1. Homebrew packages
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10 rust llvm git duckdb

# 2. Add LLVM to PATH for C++20 (Apple clang lags slightly)
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 3. Clone and enter
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials/competitions/glioblastoma-1hr-trial

# 4. Python venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# 5. Build the Rust runner
cargo build --release --manifest-path src/simulation/Cargo.toml

# 6. Build the C++ control loop
clang++ -std=c++20 -O2 -o build/robot_loop src/control/robot_loop.cpp

# 7. Run a single iteration smoke test
python -m src.simulation.iterate --seed 20260509 --iterations 1 --out data/iterations
```

## Windows Recipe (Windows 11, PowerShell 7)

```
# 1. Install winget packages
winget install Python.Python.3.10
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --quiet"
winget install Git.Git
winget install DuckDB.cli

# 2. Reopen PowerShell so PATH updates take effect, then:
git clone https://github.com/kevinkawchak/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials\competitions\glioblastoma-1hr-trial

# 3. Python venv
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .

# 4. Build the Rust runner
cargo build --release --manifest-path src/simulation/Cargo.toml

# 5. Build the C++ control loop (from a Developer PowerShell for VS 2022)
cl /std:c++20 /O2 /Fe:build\robot_loop.exe src\control\robot_loop.cpp

# 6. Run a single iteration smoke test
python -m src.simulation.iterate --seed 20260509 --iterations 1 --out data\iterations
```

## Docker Recipe (any host)

```
# 1. From the simulation directory
cd competitions/glioblastoma-1hr-trial

# 2. Bring the multi-service stack online
docker compose up --build -d

# 3. Run a single iteration inside the orchestrator container
docker compose exec orchestrator python -m src.simulation.iterate --seed 20260509 --iterations 1
```

The `docker-compose.yml` authored by future Commit 1 brings up four services: `llm` (on-prem language model), `ingest` (sensor stream consumer), `simulator` (Rust runner), `db` (DuckDB sidecar with a Parquet-mounted volume).

## Conventional High-End Server Recipe

The future session must verify that the simulation runs on conventional high-end servers without GPU acceleration. Reference profile:

- 32-core x86_64 CPU (AMD EPYC 7543 or equivalent)
- 128 GB RAM
- 2 TB NVMe SSD
- No GPU
- Linux (Ubuntu 22.04 LTS or RHEL 9)

A single iteration runs in under 90 seconds wall-clock on this profile. Sixty-four iterations run in approximately 90 minutes serial or 6 minutes with `iterate.py --jobs 16`.

## Verification

The future Commit 1 README must include the verification block below. The future session must run the block and capture its output to `logs/iteration_run.txt` before the simulation is considered runnable.

```
python -m src.simulation.iterate --seed 20260509 --iterations 1 --out data/iterations 2>&1 | tee -a logs/iteration_run.txt
test -f data/iterations/run_00001.parquet && echo "OK: iteration 1 complete"
test -f data/sensor_sample.csv && echo "OK: sensor sample present"
test -f data/xyz_trace_sample.csv && echo "OK: xyz sample present"
```

## Source Files Cited

- `requirements.txt`. Source for Python dependency baseline. The future `pyproject.toml` must be a subset compatible with Python 3.10, 3.11, and 3.12.
- `.github/workflows/ci.yml`. Source for the CI matrix that the local recipes must remain compatible with.
- `sponsor/final_paper/168_hours/instructions/core_i5_6200u_4gb/`. Source for the existing repository pattern of providing per-host runtime instructions that the future README must mirror.
- `sponsor/final_paper/168_hours/instructions/rtx_4090_openclaw/`. Same pattern, referenced for completeness.
