#!/usr/bin/env python3
"""
Script to transform dataset.json between original and simplified nested formats.
Features robust matching of crops based on content rather than just IDs.
"""
import json
import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Transform dataset.json between formats')
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output dataset JSON file')
    parser.add_argument('--mode', type=str, choices=['transform', 'revert'], default='transform',
                        help='Operation mode: transform (to nested format) or revert (to original format)')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to the original unfiltered dataset to retrieve crop region information (for revert mode)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed matching information')
    return parser.parse_args()

def transform_dataset(input_path, output_path):
    """Transform original format to nested format"""
    # Load the original dataset
    print(f"Loading dataset from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f: # Added encoding
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return


    entries = data.get('entries', [])
    if not entries:
        print("Warning: No 'entries' found in the input file.")
        # Still proceed to create an empty output if needed
        
    # Create the new structure
    transformed_data = {}
    processed_entries = 0

    print("Transforming data to nested format...")
    for entry in entries:
        # Basic check for required keys in the entry
        if not all(k in entry for k in ['original_image', 'crop_id', 'text_instances']):
            print(f"Warning: Skipping entry due to missing keys: {entry.get('crop_id', 'Unknown ID')}")
            continue

        # Extract the original image name (without "train/" prefix)
        original_image = entry['original_image']
        if isinstance(original_image, str) and original_image.startswith('train/'):
            original_image = original_image[6:]  # Remove "train/"

        # Remove file extension if present
        image_id = os.path.splitext(original_image)[0] # Use os.path.splitext

        # Extract the crop index from crop_id
        crop_id = entry['crop_id']
        # Find the crop number at the end of the crop_id
        if isinstance(crop_id, str) and '_crop_' in crop_id:
            try:
                crop_idx = crop_id.split('_crop_')[-1]
                # Optional: Validate if crop_idx is numeric if expected
                # int(crop_idx)
            except ValueError:
                 print(f"Warning: Could not parse crop index from crop_id '{crop_id}'. Using fallback.")
                 crop_idx = "0" # Fallback if parsing fails
            except Exception: # Catch other potential issues with split
                 print(f"Warning: Error processing crop_id '{crop_id}'. Using fallback.")
                 crop_idx = "0"
        else:
            # Fallback if format is different or crop_id is not a string
            print(f"Warning: Unexpected crop_id format '{crop_id}'. Using fallback index '0'.")
            crop_idx = "0"

        # --- MODIFICATION START ---
        # Extract text instances (keeping bbox, text, AND polygon if available)
        text_instances = []
        if isinstance(entry['text_instances'], list): # Check if it's a list
            for instance in entry['text_instances']:
                # Basic check for required keys in the instance
                if not isinstance(instance, dict) or not all(k in instance for k in ['bbox', 'text']):
                     print(f"Warning: Skipping instance in crop {crop_id} due to missing keys or wrong format: {instance}")
                     continue

                instance_data = {
                    'bbox': instance['bbox'],
                    'text': instance['text']
                }
                # Check for and add polygon data if it exists
                polygon_data = instance.get('polygon')
                if polygon_data is not None: # Add polygon if key exists and value is not None
                    instance_data['polygon'] = polygon_data

                text_instances.append(instance_data)
        else:
             print(f"Warning: 'text_instances' is not a list in crop {crop_id}. Skipping instances for this crop.")
        # --- MODIFICATION END ---

        # Add to transformed data structure
        if image_id not in transformed_data:
            transformed_data[image_id] = {}

        # Handle potential duplicate crop_idx for the same image_id if necessary
        if crop_idx in transformed_data[image_id]:
            print(f"Warning: Duplicate crop index '{crop_idx}' found for image '{image_id}'. Overwriting previous data.")
            
        transformed_data[image_id][crop_idx] = text_instances
        processed_entries += 1

    # Save the transformed dataset
    print(f"Saving transformed data to: {output_path}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(transformed_data, f, indent=2, ensure_ascii=False) # Added indent and ensure_ascii
    except IOError as e:
        print(f"Error: Could not write output file to {output_path}. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return

    print(f"Transformation complete. Transformed dataset saved to {output_path}")
    print(f"Original entry count processed: {processed_entries}/{len(entries)}")
    print(f"Transformed image count: {len(transformed_data)}")

    # Calculate total number of crops
    total_crops = sum(len(crops) for crops in transformed_data.values())
    print(f"Total crop count in transformed data: {total_crops}")

    # Calculate total number of text instances
    total_instances = sum(
        sum(len(instances) for instances in crops.values())
        for crops in transformed_data.values()
    )
    print(f"Total text instances in transformed data: {total_instances}")


# --- The rest of the functions (calculate_crop_similarity, normalize_image_id, revert_dataset, main) remain exactly the same ---

def calculate_crop_similarity(crop1_instances, crop2_instances):
    """
    Calculate similarity score between two crops based on their text instances

    Args:
        crop1_instances: List of text instances from first crop
        crop2_instances: List of text instances from second crop

    Returns:
        Similarity score (higher is more similar)
    """
    # If lengths are very different, they're probably not the same crop
    len_diff = abs(len(crop1_instances) - len(crop2_instances))
    if len_diff > max(3, min(len(crop1_instances), len(crop2_instances)) // 2):
        return 0

    # Count matching text instances
    matching_texts = 0
    bbox_similarity_sum = 0

    # For each instance in crop1, find best matching instance in crop2
    for inst1 in crop1_instances:
        # Ensure instance is a dictionary and has required keys
        if not isinstance(inst1, dict) or 'text' not in inst1 or 'bbox' not in inst1:
            continue
        text1 = inst1['text']
        bbox1 = inst1['bbox']
        # Basic validation for bbox format
        if not isinstance(bbox1, list) or len(bbox1) != 4:
            continue


        best_match_score = 0
        for inst2 in crop2_instances:
             # Ensure instance is a dictionary and has required keys
            if not isinstance(inst2, dict) or 'text' not in inst2 or 'bbox' not in inst2:
                continue
            text2 = inst2['text']
            bbox2 = inst2['bbox']
            # Basic validation for bbox format
            if not isinstance(bbox2, list) or len(bbox2) != 4:
                continue

            # Text must match
            if text1 != text2:
                continue

            # Calculate bbox similarity (0 to 1, higher is better)
            # Simple approach: Check if top-left coordinates are close
            try:
                x_diff = abs(bbox1[0] - bbox2[0])
                y_diff = abs(bbox1[1] - bbox2[1])
            except (TypeError, IndexError): # Handle cases where bbox elements aren't numbers or index is out of bounds
                continue


            # Normalize differences based on typical bbox size
            typical_size = 50  # Assume typical bbox dimension is around 50 pixels
            x_sim = max(0, 1 - (x_diff / typical_size))
            y_sim = max(0, 1 - (y_diff / typical_size))

            bbox_sim = (x_sim + y_sim) / 2
            match_score = bbox_sim

            if match_score > best_match_score:
                best_match_score = match_score

        if best_match_score > 0.5:  # Consider a match if similarity > 0.5
            matching_texts += 1
            bbox_similarity_sum += best_match_score

    # Calculate final similarity score
    if matching_texts == 0:
        return 0

    # Weight based on number of matching texts and their bbox similarity
    avg_bbox_similarity = bbox_similarity_sum / matching_texts if matching_texts > 0 else 0

    # Calculate percentage of matching texts (relative to smaller crop)
    # Avoid division by zero if one crop has 0 instances (though len_diff check might catch this)
    min_len = min(len(crop1_instances), len(crop2_instances))
    match_ratio = matching_texts / min_len if min_len > 0 else 0

    # Final score: combination of match ratio and bbox similarity
    similarity = 0.7 * match_ratio + 0.3 * avg_bbox_similarity

    return similarity

def normalize_image_id(image_id):
    """Normalize image ID to handle different formats"""
    # Ensure input is a string
    if not isinstance(image_id, str):
        return "" # Or raise an error, depending on desired behavior

    # Remove train/ prefix if present
    if image_id.startswith('train/'):
        image_id = image_id[6:]

    # Remove file extension if present
    image_id = os.path.splitext(image_id)[0] # Use os.path.splitext for robustness

    return image_id

def revert_dataset(input_path, output_path, reference_path=None, verbose=False):
    """
    Revert from nested format back to original format with robust crop matching

    Args:
        input_path: Path to the nested format dataset (filtered)
        output_path: Path to save the output dataset in original format
        reference_path: Path to original unfiltered dataset to retrieve crop region information
        verbose: Whether to print detailed matching information
    """
    # Load the nested dataset
    print(f"Loading nested dataset from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f: # Added encoding
            nested_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return

    # Create entries list for original format
    entries = []

    # Stats tracking
    total_crops_processed = 0
    crops_with_reference_match = 0
    images_without_reference = 0
    reference_load_failed = False

    # Index reference dataset by image ID if provided
    reference_by_image = defaultdict(list)
    if reference_path:
        print(f"Loading reference dataset from {reference_path}...")
        try:
            with open(reference_path, 'r', encoding='utf-8') as f: # Added encoding
                reference_data = json.load(f)

            # Group reference entries by image ID
            ref_entries = reference_data.get('entries', [])
            if not ref_entries:
                 print("Warning: Reference file loaded but contains no 'entries'.")

            for entry in ref_entries:
                 # Basic check for required keys in reference entry
                if not all(k in entry for k in ['original_image', 'crop_id', 'crop_region', 'text_instances']):
                    print(f"Warning: Skipping reference entry due to missing keys: {entry.get('crop_id', 'Unknown ID')}")
                    continue
                image_id = normalize_image_id(entry['original_image'])
                reference_by_image[image_id].append(entry)

            print(f"Loaded reference data for {len(reference_by_image)} unique image IDs.")
        except FileNotFoundError:
            print(f"Warning: Reference file not found at {reference_path}. Proceeding without reference data.")
            reference_path = None
            reference_load_failed = True
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to decode JSON from reference file {reference_path}: {e}. Proceeding without reference data.")
            reference_path = None
            reference_load_failed = True
        except Exception as e:
            print(f"Warning: An unexpected error occurred while loading reference dataset: {e}. Proceeding without reference data.")
            reference_path = None
            reference_load_failed = True
    else:
        print("No reference dataset path provided. Crop regions and detailed instance info will use defaults.")


    print("Reverting data to original format...")
    # Process each image in the nested data
    for image_id, crops in nested_data.items():
        # Normalize image ID from nested data key
        normalized_id = normalize_image_id(image_id)

        # Reconstruct original_image filename (assuming .jpg)
        # Use the original image_id before normalization for filename reconstruction
        original_image_filename = f"{image_id}.jpg"
        # Add train/ prefix back if it was removed during transform (heuristic)
        # A more robust way would be to store this info during transform if needed
        if not original_image_filename.startswith('train/'):
             # This assumes images were originally in a 'train/' subfolder if not specified otherwise
             # Adjust this logic if your original structure was different
            original_image_filename = f"train/{original_image_filename}"


        # Get reference crops for this image using the normalized ID
        reference_crops = reference_by_image.get(normalized_id, [])
        if reference_path and not reference_load_failed and not reference_crops:
             images_without_reference += 1
             if verbose:
                 print(f"Verbose: No reference crops found for image ID '{normalized_id}' (Original: '{image_id}').")


        # Process each crop for the current image
        if not isinstance(crops, dict):
             print(f"Warning: Expected dictionary of crops for image '{image_id}', got {type(crops)}. Skipping image.")
             continue

        for crop_idx, text_instances in crops.items():
            total_crops_processed += 1

            # Reconstruct crop_id
            # Use the reconstructed original_image_filename without extension
            base_crop_id = os.path.splitext(original_image_filename)[0]
            crop_id = f"{base_crop_id}_crop_{crop_idx}"

            # Default values for missing information
            crop_region = [0, 0, 512, 512]  # Default placeholder if no reference match
            matched_reference_entry = None # Store the best matching reference entry

            # Find matching reference crop based on content similarity
            best_match_ref_entry = None
            best_similarity = 0

            if reference_path and not reference_load_failed and reference_crops:
                # Ensure text_instances is a list before proceeding
                if not isinstance(text_instances, list):
                    print(f"Warning: text_instances for crop {crop_id} is not a list. Cannot perform matching.")
                else:
                    for ref_entry in reference_crops:
                        # Ensure reference entry has text_instances list
                        ref_text_instances = ref_entry.get('text_instances')
                        if not isinstance(ref_text_instances, list):
                            continue

                        # Prepare reference instances for comparison (only text and bbox needed)
                        ref_instances_for_comparison = []
                        for inst in ref_text_instances:
                            if isinstance(inst, dict) and 'text' in inst and 'bbox' in inst:
                                ref_instances_for_comparison.append({'text': inst['text'], 'bbox': inst['bbox']})

                        similarity = calculate_crop_similarity(
                            text_instances, # Already contains {'text': ..., 'bbox': ...}
                            ref_instances_for_comparison
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_ref_entry = ref_entry

                    # Consider it a match if similarity is above threshold
                    similarity_threshold = 0.5 # Adjust threshold if needed
                    if best_similarity > similarity_threshold:
                        crops_with_reference_match += 1
                        crop_region = best_match_ref_entry['crop_region']
                        matched_reference_entry = best_match_ref_entry # Keep track of the matched entry

                        if verbose:
                            print(f"Verbose: Matched crop {crop_id} to reference {best_match_ref_entry['crop_id']} with similarity {best_similarity:.2f}")
                    elif verbose:
                        print(f"Verbose: No good reference match found for crop {crop_id} (best similarity: {best_similarity:.2f})")

            # Create the entry for the output file
            entry = {
                "original_image": original_image_filename,
                "crop_id": crop_id,
                "crop_region": crop_region, # Use matched region or default
                "text_instances": []
            }

            # Add text instances, trying to enrich with reference data if matched
            if not isinstance(text_instances, list):
                 print(f"Warning: text_instances for crop {crop_id} is not a list. Skipping instance processing.")
            else:
                for i, instance in enumerate(text_instances):
                    # Ensure instance is a dictionary with expected keys
                    if not isinstance(instance, dict) or 'text' not in instance or 'bbox' not in instance:
                        print(f"Warning: Skipping invalid instance in crop {crop_id}: {instance}")
                        continue

                    # Default values for potentially missing fields
                    score = 1.0
                    overlaps = False
                    instance_id = f"{normalized_id}_{crop_idx}_{i}" # Generate a unique ID
                    # --- POLYGON HANDLING IN REVERT (Placeholder/Example) ---
                    # If polygons were stored in nested format, retrieve them here:
                    # polygon = instance.get('polygon') # Get polygon from nested data if it exists
                    # If polygons need to be retrieved from reference:
                    polygon_from_ref = None


                    # Try to find matching instance in the matched reference crop
                    matching_ref_instance = None
                    if matched_reference_entry:
                        ref_instances = matched_reference_entry.get('text_instances', [])
                        if isinstance(ref_instances, list):
                            text = instance['text']
                            bbox = instance['bbox']
                            best_inst_match_ref = None
                            best_inst_similarity = 0

                            for ref_instance in ref_instances:
                                if not isinstance(ref_instance, dict): continue # Skip invalid ref instances

                                # Match based on text primarily
                                if ref_instance.get('text') == text:
                                    # Secondary check: bbox similarity (optional but helpful)
                                    ref_bbox = ref_instance.get('bbox')
                                    if isinstance(ref_bbox, list) and len(ref_bbox) == 4 and isinstance(bbox, list) and len(bbox) == 4:
                                        try:
                                            x_diff = abs(ref_bbox[0] - bbox[0])
                                            y_diff = abs(ref_bbox[1] - bbox[1])
                                            # Simple similarity measure (higher is better, 1/(1+dist))
                                            similarity = 1 / (1 + x_diff + y_diff)
                                        except (TypeError, IndexError):
                                            similarity = 0 # Handle potential errors in bbox data
                                    else:
                                        similarity = 0.1 # Low similarity if bboxes can't be compared but text matches

                                    if similarity > best_inst_similarity:
                                        best_inst_similarity = similarity
                                        best_inst_match_ref = ref_instance

                            # Consider it a match if similarity is good enough or text matched
                            match_threshold = 0.01 # Low threshold, primarily relies on text match
                            if best_inst_similarity > match_threshold:
                                matching_ref_instance = best_inst_match_ref


                    # Use reference instance values if available, otherwise use placeholders/defaults
                    if matching_ref_instance:
                        score = matching_ref_instance.get("score", 1.0)
                        overlaps = matching_ref_instance.get("overlaps", False)
                        instance_id = matching_ref_instance.get("id", instance_id) # Prefer reference ID
                        polygon_from_ref = matching_ref_instance.get("polygon") # Get polygon from reference


                    # Construct the final text instance for the output
                    text_instance_output = {
                        "bbox": instance["bbox"],  # Keep the bbox from the input nested data
                        "text": instance["text"],  # Keep the text from the input nested data
                        "score": score,
                        "overlaps": overlaps,
                        "id": instance_id
                    }

                    # --- ADDING POLYGON TO REVERTED OUTPUT ---
                    # Decide which polygon to use (from nested input or from reference)
                    # Example: Prioritize polygon from reference if available
                    final_polygon = polygon_from_ref # if polygon_from_ref is not None else polygon
                    if final_polygon is not None:
                         text_instance_output["polygon"] = final_polygon


                    entry["text_instances"].append(text_instance_output)

            entries.append(entry)

    # Create final output format
    output_data = {"entries": entries}

    # Save the original format dataset
    print(f"Saving reverted data to: {output_path}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(output_data, f, indent=2, ensure_ascii=False) # Added indent and ensure_ascii
    except IOError as e:
        print(f"Error: Could not write output file to {output_path}. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return


    print(f"Reversion complete. Reverted dataset saved to {output_path}")
    print(f"Total images processed from nested format: {len(nested_data)}")
    print(f"Total entries created in reverted format: {len(entries)}")
    print(f"Total crops processed: {total_crops_processed}")

    if reference_path and not reference_load_failed:
        match_percentage = (crops_with_reference_match / total_crops_processed * 100) if total_crops_processed > 0 else 0
        print(f"Reference matching: {crops_with_reference_match}/{total_crops_processed} crops matched ({match_percentage:.1f}%)")
        if images_without_reference > 0:
             print(f"Images present in input but not found in reference: {images_without_reference}")
    elif reference_load_failed:
         print("Reference data loading failed. Crop regions and detailed instance info use defaults.")
    else:
         print("No reference data used. Crop regions and detailed instance info use defaults.")


    # Calculate total number of text instances in the final output
    total_instances_output = sum(len(entry.get("text_instances", [])) for entry in entries)
    print(f"Total text instances in reverted output: {total_instances_output}")


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    if args.mode == 'transform':
        transform_dataset(args.input, args.output)
    else:  # args.mode == 'revert'
        # Check if reference is needed and provided for revert mode
        if not args.reference:
             print("Warning: Revert mode typically benefits from a --reference file to restore crop regions and instance details. Proceeding without it.")
        elif not os.path.exists(args.reference):
             print(f"Warning: Reference file specified but not found at {args.reference}. Proceeding without reference data.")
             args.reference = None # Treat as if no reference was provided

        revert_dataset(args.input, args.output, args.reference, args.verbose)

if __name__ == '__main__':
    main()