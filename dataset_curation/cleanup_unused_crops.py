import os
import json
import argparse
import logging
import sys
from pathlib import Path

# Basic logging setup for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def cleanup_crops(json_file_path, dry_run=True):
    """
    Deletes image files in the 'cropped_images' directory that are not
    referenced in the provided dataset JSON file.

    Args:
        json_file_path (str): Path to the final dataset JSON file
                              (e.g., full_dataset_SUFFIX.json).
        dry_run (bool): If True, only print actions without deleting files.
                        If False, perform actual deletion.
    """
    logging.info(f"Starting crop cleanup based on: {json_file_path}")
    if dry_run:
        logging.warning("--- DRY RUN MODE ENABLED: No files will be deleted. ---")

    # --- Validate JSON file path ---
    json_path = Path(json_file_path)
    if not json_path.is_file():
        logging.error(f"Error: JSON file not found at '{json_file_path}'")
        return False

    # --- Determine crop images directory path ---
    # Assumes 'cropped_images' is a sibling directory to the JSON file's parent directory
    # e.g., if JSON is /path/to/outputs/final_dataset_SUFFIX/dataset.json
    # then crops are expected in /path/to/outputs/cropped_images/
    # Adjust this logic if your structure is different.
    # Example: json parent is final_dataset_SUFFIX, its parent is the base output dir
    base_output_dir = json_path.parent
    crop_dir_path = base_output_dir / "cropped_images"

    logging.info(f"Expecting cropped images in: {crop_dir_path}")
    if not crop_dir_path.is_dir():
        logging.error(f"Error: Cropped images directory not found at '{crop_dir_path}'")
        logging.error("Please ensure the 'cropped_images' directory exists relative to the JSON file's location.")
        return False

    # --- Read JSON and extract valid filenames ---
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "entries" not in data:
            logging.error(f"Error: JSON file '{json_file_path}' missing 'entries' key.")
            return False

        # The 'original_image' field in the formatted dataset holds the crop filename
        valid_filenames = set(entry.get("original_image") for entry in data["entries"] if entry.get("original_image"))
        logging.info(f"Found {len(valid_filenames)} valid crop filenames referenced in the JSON.")
        if not valid_filenames:
             logging.warning("JSON file contains no valid image references. No files will be kept.")

    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from '{json_file_path}'.")
        return False
    except Exception as e:
        logging.error(f"Error reading JSON file '{json_file_path}': {e}")
        return False

    # --- Scan directory and compare ---
    deleted_count = 0
    kept_count = 0
    error_count = 0
    scanned_files = 0

    logging.info(f"Scanning directory '{crop_dir_path}' for cleanup...")
    try:
        # Using scandir for potentially better performance on large directories
        for item in os.scandir(crop_dir_path):
            if item.is_file():
                scanned_files += 1
                filename = item.name
                # Check if this filename is in the set of valid names from JSON
                if filename in valid_filenames:
                    kept_count += 1
                    logging.debug(f"Keeping: {filename}")
                else:
                    # File exists in directory but not in JSON - mark for deletion
                    file_to_delete = Path(item.path)
                    logging.info(f"Marked for deletion: {filename}")
                    if not dry_run:
                        try:
                            os.remove(file_to_delete)
                            deleted_count += 1
                            logging.info(f"  -> DELETED: {filename}")
                        except OSError as e:
                            logging.error(f"  -> FAILED to delete {filename}: {e}")
                            error_count += 1
                    else:
                        # In dry run, just increment the count as if deleted
                        deleted_count += 1

            # Optionally log directories found, but typically not needed
            # elif item.is_dir():
            #     logging.debug(f"Skipping directory: {item.name}")

    except Exception as e:
        logging.error(f"An error occurred while scanning directory '{crop_dir_path}': {e}")
        return False

    # --- Final Summary ---
    logging.info("-----------------------------------------")
    logging.info("Cleanup Scan Summary:")
    logging.info(f"  Directory Scanned: {crop_dir_path}")
    logging.info(f"  JSON File Used:    {json_file_path}")
    logging.info(f"  Files Scanned:     {scanned_files}")
    logging.info(f"  Files Kept:        {kept_count} (referenced in JSON)")
    if dry_run:
        logging.info(f"  Files To Be Deleted: {deleted_count}")
        logging.warning("--- DRY RUN MODE: No files were actually deleted. ---")
    else:
        logging.info(f"  Files Deleted:     {deleted_count}")
        if error_count > 0:
            logging.error(f"  Deletion Errors:   {error_count}")
    logging.info("-----------------------------------------")

    return error_count == 0 # Return True if successful (no errors during deletion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up unused cropped images based on a dataset JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example Usage:
  Dry Run (Recommended First):
    python cleanup_unused_crops.py --json /path/to/outputs/final_dataset_SUFFIX/restoration_dataset_SUFFIX.json --dry-run

  Actual Deletion (Use with caution!):
    python cleanup_unused_crops.py --json /path/to/outputs/final_dataset_SUFFIX/restoration_dataset_SUFFIX.json
"""
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the final dataset JSON file (e.g., restoration_dataset_SUFFIX.json or full_dataset_SUFFIX.json) "
             "that contains the list of valid crop filenames."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run. List files that would be deleted without actually deleting them."
    )
    args = parser.parse_args()

    success = cleanup_crops(args.json, args.dry_run)

    if not success and not args.dry_run:
        sys.exit(1) # Exit with error code if deletion failed
    sys.exit(0) # Exit successfully