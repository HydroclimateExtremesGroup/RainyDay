#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_rainyday.sh [ENV_NAME] path/to/params.json
# If ENV_NAME is omitted it defaults to 'RainyDay_Env'.

ENV_NAME="${1:-RainyDay_Env}"
PARAM_FILE="${2:-}"

if [ -z "$PARAM_FILE" ]; then
  echo "Usage: $0 [ENV_NAME] path/to/params.json"
  echo "Example: $0 RainyDay_Env Examples/BigThompson/BigThompsonExample.json"
  exit 1
fi

# Try common conda locations and fall back to PATH
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # conda is on PATH (shell integration may already be present)
  :
else
  echo "conda not found. Please install Miniconda/Mambaforge or run 'conda init bash' and restart your shell."
  exit 1
fi

# Activate the environment
conda activate "$ENV_NAME" || {
  echo "Failed to activate conda env '$ENV_NAME'. Make sure it exists (conda env list)."
  exit 1
}

# Run RainyDay
python Source/RainyDay_Py3.py "$PARAM_FILE"
