# RainyDay Local Docker Build Guide
## For Non-Compatible Architectures

This guide is for users whose system architecture is not compatible with the pre-built Docker Hub image, or who prefer to build the image locally.

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

---

## Required Files Structure

Ensure you have the following files in your RainyDay directory:

```
RainyDay/
├── Dockerfile                   # Docker build instructions
├── RainyDay_Env.yml            # Conda environment file
├── Source/                     # Source code directory
│   ├── RainyDay_Py3.py        # Main application
│   └── RainyDay_utilities_Py3/
│       └── RainyDay_functions.py
└── [your output directory]/    # Will be created
    └── params.json             # Your parameter file
```

---

## Build and Run Commands

### Step 1: Build the Docker Image
```bash
docker build -f Dockerfile -t rainyday_img:latest .
```

### Step 2: Create Host Output Directory
```bash
# Create output directory (Docker will create with root permissions if not pre-created)
mkdir -p path/to/output
```

### Step 3: Run the Container
```bash
docker run -v path/to/output:/output rainyday_img:latest params
```

**Note**: Replace `params` with your actual parameter file path (e.g., `/output/your_params.json`)

---

## Docker Container Structure

### Internal Container Layout
```
Container Directory Structure:
/
├── RainyDay_Py3.py              # Main application entry point
├── RainyDay_utilities_Py3/
│   └── RainyDay_functions.py    # Utility functions
├── RainyDay_Env.yml             # Conda environment configuration
└── output/                      # Your mounted directory (volume)
    ├── params.json              # Your parameter file
    └── [generated files]        # Output files created by RainyDay
```

### Host-Container Directory Link

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `path/to/output` | `/output` | Bidirectional directory mapping |

**Volume Mapping**: `-v path/to/output:/output`
- **Host side**: `path/to/output` (on your local computer)
- **Container side**: `/output` (inside Docker container)
- **Result**: Any files written to `/output` in the container appear in `path/to/output` on your machine

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
- `"MAINPATH": "path/to/output"` (host path)
- `"MAINPATH": "./output"` (relative path)

### ✅ Correct Container Path:
- `"MAINPATH": "/output"` (container internal path)

---

## Troubleshooting

- **Docker not running:** Start Docker Desktop or Docker daemon
- **Build fails:** Check that all required files are present
- **Permission issues:** The script creates `./output` with proper permissions
- **Missing Dockerfile:** Ensure you're running the script from the RainyDay directory