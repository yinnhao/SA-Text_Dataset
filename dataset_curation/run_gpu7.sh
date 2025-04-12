#!/bin/bash
# Runs pipeline sequentially for subsets assigned to GPU 7

# --- Configuration ---
CONFIG_FILE="config.yaml"
PYTHON_SCRIPT="main_pipeline.py"
CONDA_ENV_NAME="hyunbin"
GPU_ID="7" # GPU assigned to this script
SUBSETS_TO_PROCESS=("sa_000013" "sa_000015" "sa_000017") # Subsets for this GPU

# --- Activate Conda Environment ---
echo "Activating conda environment: $CONDA_ENV_NAME for GPU $GPU_ID"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    exit 1
fi
echo "Conda environment activated."

# --- Process Subsets Sequentially ---
overall_success=true
total_subsets=${#SUBSETS_TO_PROCESS[@]}
current_subset_num=0

for subset in "${SUBSETS_TO_PROCESS[@]}"; do
  current_subset_num=$((current_subset_num + 1))
  output_suffix=${subset} # Use subset name as suffix

  echo "#####################################################"
  echo "Starting Pipeline for Subset: $subset ($current_subset_num/$total_subsets) on GPU: $GPU_ID"
  echo "Output Suffix: $output_suffix"
  echo "#####################################################"

  # Run the command in the foreground (wait for it to finish)
  CUDA_VISIBLE_DEVICES=$GPU_ID python "$PYTHON_SCRIPT" \
    --config "$CONFIG_FILE" \
    --sa1b_subfolder "$subset" \
    --output_suffix "$output_suffix"

  EXIT_CODE=$? # Capture exit code

  if [ $EXIT_CODE -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Pipeline for Subset: $subset FAILED with exit code $EXIT_CODE."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    overall_success=false # Mark that at least one job failed
    # Decide whether to continue or exit on failure
    # exit $EXIT_CODE # Uncomment this line to stop the script immediately on failure
  else
    echo "-----------------------------------------------------"
    echo "Pipeline for Subset: $subset completed successfully."
    echo "-----------------------------------------------------"
  fi
  echo # Add a blank line for readability between subsets
done

# --- Final Summary ---
echo "#####################################################"
if $overall_success; then
  echo "All assigned subsets for GPU $GPU_ID completed successfully."
  exit 0
else
  echo "One or more subsets failed for GPU $GPU_ID."
  exit 1
fi
echo "#####################################################"

# Optional: Deactivate environment
# conda deactivate