# RainyDay Docker Runner (Docker Hub)
## For Compatible Architectures

This guide helps you pull and run the pre-built RainyDay application from Docker Hub.

---

## Prerequisites

### 1. Create Python Virtual Environment (Optional but Recommended)
```bash
# Create a virtual environment
python3 -m venv rainyday_env

# Activate the environment
# On Linux/Mac:
source rainyday_env/bin/activate
# On Windows:
rainyday_env\Scripts\activate
```

### 2. Install Docker
- **Windows/Mac**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Install Docker Engine using your package manager:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io

# CentOS/RHEL
sudo yum install docker
```

### 3. Ensure Docker Engine is Running
- **Windows/Mac**: Start Docker Desktop application
- **Linux**: Start Docker daemon:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### 4. Verify Docker Installation
```bash
docker --version
docker info
```

### 5. Compatible Architecture Check
This Docker image works on:
- **linux/amd64** (Intel/AMD 64-bit processors)
- **linux/arm64** (Apple Silicon M1/M2, ARM64 processors)

If you have compatibility issues, use the [local build guide](README_LocalBuild.md) instead.

---

## Quick Start Commands

### Step 1: Pull the RainyDay Image
```bash
docker pull adityagoyal333/rainyday:rainyday_img
```

### Step 2: Create Host Output Directory
```bash
# Create output directory on your local machine
mkdir -p path/to/output
```

### Step 3: Run the Container
```bash
docker run -v path/to/output:/output adityagoyal333/rainyday:rainyday_img params
```

**Note**: Replace `params` with your actual parameter file path (e.g., `/output/your_params.json`) and other params that you need to list

---

## Docker Container Structure

### Internal Container Layout
```
Container Directory Structure:
/
├── RainyDay_Py3.py              # Main application entry point
├── RainyDay_utilities_Py3/
│   └── RainyDay_functions.py    # Utility functions and dependencies
├── RainyDay_Env.yml             # Conda environment configuration
└── output/                      # Your mounted directory (volume)
    ├── params.json              # Your parameter file
    └── [generated files]        # Output files created by RainyDay
```

### Host-Container Directory Link

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `path/to/output` | `/output` | Bidirectional directory mapping |

**Volume Mapping**: `-v /host/output:/output`
- **Host side**: `/host/output` (on your local computer)
- **Container side**: `/output` (inside Docker container)
- **Result**: Files written to `/output` in container appear in `/host/output` on your machine

---

## Parameter File Configuration

### Critical Path Settings
Your JSON parameter file **must use container paths**:

```json
{
  "MAINPATH": "/output",           # Always use container path
  "SCENARIONAME": "my_scenario",
  "CREATECATALOG": "true",
  "CATALOGNAME": "my_catalog",
  "VARIABLES": ["precipitation", "latitude", "longitude"]
}
```

### ❌ Incorrect Paths:
- `"MAINPATH": "/host/output"` (host path)
- `"MAINPATH": "./output"` (relative path)

### ✅ Correct Container Path:
- `"MAINPATH": "/output"` (container internal path)

---
