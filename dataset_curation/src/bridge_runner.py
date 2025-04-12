import subprocess
import os
import logging
import shlex
import time

from .utils import ensure_dir


def run_bridge(config, input_dir, output_dir, stage1=True):
    """
    Constructs and runs the Bridge inference command using subprocess.
    """
    ensure_dir(output_dir)
    output_json_path = os.path.join(output_dir, "text_detection_results.json")
    bridge_repo_dir = config['bridge_repo_dir']
    inference_script = os.path.join(bridge_repo_dir, "demo/inference.py")
    config_file = os.path.join(bridge_repo_dir, config['bridge_config_file'])
    weights_file = os.path.join(bridge_repo_dir, config['bridge_weights_file'])
    confidence = config['bridge_confidence_threshold']
    python_cmd_str = config.get('bridge_env_python', 'python')

    try:
        python_cmd_parts = shlex.split(python_cmd_str)
    except ValueError as e:
        logging.error(f"Error parsing bridge_env_python command '{python_cmd_str}': {e}.")
        return None

    base_cmd = python_cmd_parts + [
        inference_script, "--config-file", config_file, "--input", input_dir,
        "--output", output_dir, "--confidence-threshold", str(confidence),
        "--no-visualization",
    ]
    if not stage1:
        base_cmd.append("--use-polygon")
    opts_cmd = ["--opts", "MODEL.WEIGHTS", weights_file, "MODEL.TRANSFORMER.USE_BOX", "True"]
    cmd = base_cmd + opts_cmd
    cmd = [str(c) for c in cmd]

    # logging.info(f"Running Bridge {'Stage 1' if stage1 else 'Stage 2'}...") # Included in time_step
    logging.info(f"Command: {shlex.join(cmd)}") # Keep command log

    try:
        result = subprocess.run(
            cmd, cwd=bridge_repo_dir, check=True, capture_output=True,
            text=True, encoding='utf-8'
        )
        # Only log stdout/stderr if there's an error or if debug level is enabled
        if result.returncode != 0 or logging.getLogger().isEnabledFor(logging.DEBUG):
            stdout_log = result.stdout.strip()
            stderr_log = result.stderr.strip()
            if len(stdout_log) > 1000: stdout_log = "[...] " + stdout_log[-1000:]
            if len(stderr_log) > 1000: stderr_log = "[...] " + stderr_log[-1000:]
            if stdout_log: logging.debug(f"Bridge stdout:\n{stdout_log}") # Log as DEBUG
            if stderr_log: logging.warning(f"Bridge stderr:\n{stderr_log}") # Log stderr as WARNING

        if os.path.exists(output_json_path):
            # logging.info(f"Bridge {'Stage 1' if stage1 else 'Stage 2'} completed. Output: {output_json_path}") # Included in time_step
            return output_json_path
        else:
            time.sleep(0.5) # Check again after delay
            if os.path.exists(output_json_path):
                # logging.info(f"Bridge {'Stage 1' if stage1 else 'Stage 2'} completed (after delay). Output: {output_json_path}") # Included in time_step
                return output_json_path
            else:
                logging.error(f"Bridge command ran but output file not found: {output_json_path}")
                try:
                    dir_contents = os.listdir(output_dir)
                    logging.error(f"Contents of {output_dir}: {dir_contents}")
                except Exception as list_e:
                    logging.error(f"Could not list contents of {output_dir}: {list_e}")
                return None
    except FileNotFoundError:
        logging.error(f"Error: Command '{cmd[0]}' not found. Check 'bridge_env_python' in config ('{python_cmd_str}') or PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Bridge execution failed with return code {e.returncode}")
        logging.error(f"Stderr:\n{e.stderr}") # Keep error details
        logging.error(f"Stdout:\n{e.stdout}") # Keep error details
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while running Bridge: {e}", exc_info=True)
        return None