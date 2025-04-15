import argparse
import yaml
import os
import logging
import sys
import time

# Import necessary modules from your pipeline structure
from src import utils
from src import cropping

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate cropped images based on Stage 1 detections.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to the main configuration YAML file")
    parser.add_argument("--stage1_json", required=True, help="Path to the EXISTING Bridge Stage 1 results JSON file for the target subset.")
    parser.add_argument("--sa1b_subfolder", required=True, type=str, help="The specific SA-1B subfolder to process (e.g., 'sa_000012')")
    parser.add_argument("--output_suffix", required=True, type=str, help="The EXACT output suffix used for the original run (e.g., 'sa_000012')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for consistency if cropping algorithm uses random elements)")

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded base configuration from: {args.config}")
    except Exception as e:
        print(f"FATAL: Error loading config file {args.config}: {e}")
        sys.exit(1)

    # --- Define Paths ---
    # Override subfolder in config for input path calculation
    config['sa1b_subfolder'] = args.sa1b_subfolder
    sa1b_input_dir = os.path.join(config['sa1b_base_dir'], config['sa1b_subfolder'])

    # Construct the specific output directory for cropped images for this run
    # Use the provided output_suffix directly
    intermediate_base = config['intermediate_output_base_dir'] + args.output_suffix
    crop_image_dir = os.path.join(intermediate_base, "cropped_images")

    # --- Setup Logging ---
    # Log to console only for this simple script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # --- Set Seed ---
    utils.seed_everything(args.seed) # Use the utils function

    logging.info(f"Starting Re-Cropping for Subset: {args.sa1b_subfolder}")
    logging.info(f"Using Stage 1 JSON: {args.stage1_json}")
    logging.info(f"Original Images Dir: {sa1b_input_dir}")
    logging.info(f"Output Crop Dir: {crop_image_dir}")

    # --- Ensure Input/Output Dirs Exist ---
    if not os.path.exists(args.stage1_json):
        logging.error(f"Stage 1 JSON not found: {args.stage1_json}")
        sys.exit(1)
    if not os.path.isdir(sa1b_input_dir):
        logging.error(f"SA-1B input directory not found: {sa1b_input_dir}")
        sys.exit(1)
    utils.ensure_dir(crop_image_dir) # Create output dir if needed

    start_time = time.time()

    try:
        # --- Step 2: Define Crop Regions (using existing Stage 1 results) ---
        logging.info("--- Defining Crop Regions ---")
        crop_definitions = cropping.define_crop_regions(args.stage1_json, config)
        if not crop_definitions:
             # Check if stage1_json was empty or just didn't yield crops
             stage1_data = utils.read_json(args.stage1_json)
             if stage1_data and not stage1_data.get("annotations"):
                 logging.warning("Stage 1 JSON contained no annotations. No crops will be generated.")
             else:
                 # Raise error if definitions failed for other reasons
                 raise RuntimeError("Crop definition failed to produce any results.")
        logging.info(f"--- Finished Defining Crop Regions (Duration: {time.time() - start_time:.2f}s) ---")


        # --- Step 3: Create Crop Images ---
        step3_start_time = time.time()
        logging.info("--- Creating Crop Images ---")
        cropping.create_crop_images(crop_definitions, sa1b_input_dir, crop_image_dir, config)
        logging.info(f"--- Finished Creating Crop Images (Duration: {time.time() - step3_start_time:.2f}s) ---")

        logging.info(f"Re-Cropping for {args.sa1b_subfolder} completed successfully.")

    except Exception as e:
        logging.error(f"Re-Cropping failed for {args.sa1b_subfolder}: {e}", exc_info=True)
        sys.exit(1)

    finally:
        end_time = time.time()
        logging.info(f"Total Re-Cropping time for {args.sa1b_subfolder}: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()