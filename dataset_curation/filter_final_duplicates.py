import json
import os
import argparse
import logging
import sys
from collections import defaultdict
import copy
from pathlib import Path
from tqdm import tqdm

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- ADDED: ensure_dir function ---
def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary."""
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        except OSError as e:
            logging.error(f"Error creating directory {path}: {e}")
            raise # Re-raise the error if directory creation fails critically
# --- END ADDED ---


# --- Helper: Calculate Overlap ---
def calculate_overlap(region1, region2):
    """Calculate the IoU of two regions [x1, y1, x2, y2]."""
    x1 = max(region1[0], region2[0])
    y1 = max(region1[1], region2[1])
    x2 = min(region1[2], region2[2])
    y2 = min(region1[3], region2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (region1[2] - region1[0]) * (region1[3] - region1[1])
    area2 = (region2[2] - region2[0]) * (region2[3] - region2[1])
    if area1 <= 0 or area2 <= 0: return 0.0
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# --- Main Filtering Function ---
def filter_duplicates_in_dataset(input_json_path, output_json_path, iou_threshold=0.9, dry_run=False):
    """
    Filters duplicate text instances within each entry of a final dataset JSON file.
    """
    logging.info(f"Starting duplicate filtering for: {input_json_path}")
    logging.info(f"IoU Threshold: {iou_threshold}")
    if dry_run:
        logging.warning("--- DRY RUN MODE ENABLED: No output file will be written. ---")

    # --- Validate JSON file path ---
    input_path = Path(input_json_path)
    if not input_path.is_file():
        logging.error(f"Error: Input JSON file not found at '{input_json_path}'")
        return False

    # --- Read JSON and extract valid filenames ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "entries" not in data or not isinstance(data["entries"], list):
            logging.error(f"Error: Input JSON file '{input_json_path}' missing 'entries' list or has incorrect format.")
            return False
        logging.info(f"Read {len(data['entries'])} entries from input file.")
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from '{input_json_path}'.")
        return False
    except Exception as e:
        logging.error(f"Error reading JSON file '{input_json_path}': {e}")
        return False

    # --- Process Entries ---
    filtered_entries = []
    total_duplicates_removed = 0
    total_instances_before = 0
    total_instances_after = 0

    for entry in tqdm(data["entries"], desc="Filtering duplicates per entry"):
        original_instances = entry.get("text_instances", [])
        n = len(original_instances)
        total_instances_before += n

        if n <= 1:
            filtered_entries.append(copy.deepcopy(entry))
            total_instances_after += n
            continue

        try:
            sorted_instances = sorted(original_instances, key=lambda x: x.get('score', 0.0), reverse=True)
        except Exception as sort_e:
             logging.error(f"Error sorting instances in entry {entry.get('crop_id', 'Unknown')}: {sort_e}. Skipping entry.")
             continue

        kept_indices = set(range(n))
        entry_duplicates_removed = 0

        for i in range(n):
            if i not in kept_indices: continue
            bbox1 = sorted_instances[i].get("bbox")
            if not (isinstance(bbox1, list) and len(bbox1) == 4):
                logging.warning(f"Instance {sorted_instances[i].get('id', 'N/A')} in {entry.get('crop_id')} has invalid bbox. Cannot check for duplicates.")
                continue

            for j in range(i + 1, n):
                if j not in kept_indices: continue
                bbox2 = sorted_instances[j].get("bbox")
                if not (isinstance(bbox2, list) and len(bbox2) == 4):
                    logging.warning(f"Instance {sorted_instances[j].get('id', 'N/A')} in {entry.get('crop_id')} has invalid bbox. Skipping comparison.")
                    continue

                overlap = calculate_overlap(bbox1, bbox2)
                if overlap > iou_threshold:
                    kept_indices.remove(j)
                    entry_duplicates_removed += 1
                    if dry_run:
                         inst_i = sorted_instances[i]; inst_j = sorted_instances[j]
                         logging.info(f"[Dry Run] Would remove instance id={inst_j.get('id','N/A')} (score={inst_j.get('score',0):.3f}) "
                                      f"due to overlap {overlap:.3f} with id={inst_i.get('id','N/A')} (score={inst_i.get('score',0):.3f}) "
                                      f"in crop {entry.get('crop_id')}")

        new_entry = copy.deepcopy(entry)
        new_entry["text_instances"] = [sorted_instances[idx] for idx in sorted(list(kept_indices))]
        filtered_entries.append(new_entry)
        total_duplicates_removed += entry_duplicates_removed
        total_instances_after += len(new_entry["text_instances"])

    # --- Write Output JSON (if not dry run) ---
    output_data = {"entries": filtered_entries}
    success = True
    if not dry_run:
        try:
            output_path = Path(output_json_path)
            # --- Use the ensure_dir function defined above ---
            ensure_dir(output_path.parent)
            # --- End change ---
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Filtered dataset saved to: {output_json_path}")
        except Exception as e:
            logging.error(f"Error writing output JSON file '{output_json_path}': {e}")
            success = False

    # --- Final Summary ---
    logging.info("-----------------------------------------")
    logging.info("Duplicate Filtering Summary:")
    logging.info(f"  Input File:           {input_json_path}")
    logging.info(f"  Entries Processed:    {len(data['entries'])}")
    logging.info(f"  Total Instances Before: {total_instances_before}")
    logging.info(f"  Duplicates Removed:   {total_duplicates_removed}")
    logging.info(f"  Total Instances After:  {total_instances_after}")
    if dry_run:
        logging.warning("--- DRY RUN MODE: No output file was written. ---")
    else:
        logging.info(f"  Output File:          {output_json_path if success else 'Write Failed'}")
    logging.info("-----------------------------------------")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter duplicate text instances from a final dataset JSON based on IoU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example Usage:
  Dry Run (Recommended First):
    python filter_final_duplicates.py --input_json /path/to/outputs/final_dataset_SUFFIX/restoration_dataset_SUFFIX.json --dry-run

  Actual Deletion (Use with caution!):
    python filter_final_duplicates.py --input_json /path/to/outputs/final_dataset_SUFFIX/restoration_dataset_SUFFIX.json
"""
    )
    parser.add_argument("--input_json", required=True, help="Path to the input dataset JSON file")
    parser.add_argument("--output_json", default=None, help="Path to save the filtered output JSON file. Defaults to appending '_no_duplicates'.")
    parser.add_argument("--iou_threshold", type=float, default=0.9, help="IoU threshold for considering instances as duplicates (default: 0.9).")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run.")
    args = parser.parse_args()

    output_path = args.output_json
    if output_path is None:
        in_path = Path(args.input_json)
        output_path = in_path.parent / f"{in_path.stem}_no_duplicates{in_path.suffix}"

    success = filter_duplicates_in_dataset(
        args.input_json,
        str(output_path),
        args.iou_threshold,
        args.dry_run
    )

    if not success and not args.dry_run:
        sys.exit(1)
    sys.exit(0)