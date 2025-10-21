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

### Host Directory Structure
Organize your files on your local machine as follows:

```
RainyDay/
├── Dockerfile                   # Docker build instructions
├── RainyDay_Env.yml            # Conda environment file
├── Source/                     # Source code directory
│   ├── RainyDay_Py3.py        # Main application
│   └── RainyDay_utilities_Py3/
│       └── RainyDay_functions.py
├── input/                 # CREATE THIS - Input files directory
│   ├── precipitation_files/    # Your .nc files go here
│   │   ├── AORC.19790201.precip.nc
│   │   ├── AORC.19790202.precip.nc
│   │   └── ...
│   └── params.json            # Your parameter configuration file
└── output/                    # CREATE THIS - Results will appear here
```

### Step 1: Set Up Input Data
```bash
# Create required directories
mkdir -p input/precipitation_files
mkdir -p output

# Place your .nc precipitation files in input/precipitation_files/
# Place your params.json in input/
```

---

## Build and Run Commands

### Step 1: Build the Docker Image
```bash
docker build -f Dockerfile -t rainyday_img:latest .
```

### Step 2: Run the Container with Volume Mounts
```bash
docker run \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  rainyday_img:latest /input/params.json
```

**Important**: The container can **only access files through volume mounts**. It cannot see files elsewhere on your system.

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
├── input/                       # Your input data (mounted)
│   ├── precipitation_files/     # Your .nc files
│   │   ├── AORC.19790201.precip.nc
│   │   └── ...
│   └── params.json              # Your parameter file
└── output/                      # Your output directory (mounted)
    └── [generated files]        # Results created by RainyDay
```

### Host-Container Directory Mapping

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `$(pwd)/input` | `/input` | Input files and parameters |
| `$(pwd)/output` | `/output` | Generated results and outputs |

**Volume Mapping**: 
- `-v $(pwd)/input:/input` - Maps your local input folder to container's /input
- `-v $(pwd)/output:/output` - Maps your local output folder to container's /output

---

## Parameter File Configuration

### Critical Path Settings
Your `params.json` file **must use container paths**, not host paths:

```json
{
  "MAINPATH": "/output",                           # Container output path
  "RAINPATH": "/input/precipitation_files",       # Container input path for .nc files
  "SCENARIONAME": "my_scenario",
  "CATALOGNAME": "my_catalog",
  "CREATECATALOG": "true",
  "DURATION": 24,
  "NYEARS": 500,
  "NSTORMS": 40,
  "NREALIZATIONS": 100,
  "SCENARIOS": "false",
  "TIMESEPARATION": 12,
  "DOMAINTYPE": "rectangular",
  "AREA_EXTENT": {
    "LATITUDE_MIN": 42.5,
    "LATITUDE_MAX": 44.0,
    "LONGITUDE_MIN": -90.5,
    "LONGITUDE_MAX": -88.0
  },
  "DIAGNOSTICPLOTS": "false",
  "FREQANALYSIS": "true",
  "EXCLUDESTORMS": "none",
  "INCLUDEYEARS": "all",
  "EXCLUDEMONTHS": [1, 2, 3, 12],
  "DURATIONCORRECTION": "true",
  "TRANSPOSITION": "uniform",
  "RESAMPLING": "poisson",
  "RETURNLEVELS": [2, 5, 10, 25, 50, 100, 200, 500],
  "POINTAREA": "grid",
  "POINTLAT": 43.075,
  "POINTLON": -89.385,
  "BOX_YMIN": 42.75,
  "BOX_YMAX": 43.0,
  "BOX_XMIN": -89.5,
  "BOX_XMAX": -89.25,
  "CALCTYPE": "pds",
  "VARIABLES": {
    "rainname": "precipitation",
    "latname": "latitude",
    "longname": "longitude"
  }
}
```

### ❌ Incorrect Paths (Host Paths):
```json
{
  "MAINPATH": "/Users/your_username/Desktop/output",     # Wrong - host path
  "RAINPATH": "./input/precipitation_files",       # Wrong - relative path
  "RAINPATH": "/absolute/host/path/to/files"            # Wrong - host path
}
```

### ✅ Correct Paths (Container Paths):
```json
{
  "MAINPATH": "/output",                                 # Correct - container path
  "RAINPATH": "/input/precipitation_files",             # Correct - container path
}
```

---

## Troubleshooting
- **Docker not running:** Start Docker Desktop or Docker daemon
- **Build fails:** Check that all required files are present
- **Permission issues:** The script creates `./output` with proper permissions
- **Missing Dockerfile:** Ensure you're running the script from the RainyDay directory