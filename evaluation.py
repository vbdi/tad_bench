import json
import os
from collections import defaultdict
import argparse
import re

from datasets import load_dataset, load_from_disk, DatasetDict


def load_json_file(filepath):
    """Loads a standard JSON file (a single JSON object/array)."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def load_jsonl_file(filepath):
    """Loads a JSONL file (one JSON object per line)."""
    data = []
    try:
        if(os.path.isfile(filepath)):     # KC
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip(): # Avoid empty lines
                        data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from a line in {filepath}: {e}")
        return None

def clean_text(text):
    """Converts text to lowercase, strips whitespace, and removes punctuation."""
    # Convert to string, strip whitespace, convert to lowercase, and remove non-word/space characters
    return re.sub(r'[^\w\s]', '', str(text).strip().lower())

def evaluate_accuracy(predictions, gt_map):
    """Calculates accuracy. Normalizes predictions to lowercase strings."""
    if not gt_map: return 0.0
    correct_count = sum(1 for item in predictions
                        if item['idx'] in gt_map and
                        str(item['pred']).lower() == str(gt_map[item['idx']]).lower())
    return correct_count / len(gt_map)

def evaluate_exact_match(predictions, gt_map):
    """
    Calculates exact match accuracy after fully cleaning and normalizing text.
    This cleaning process involves converting to lowercase, stripping whitespace,
    and removing all punctuation before comparison.
    """
    if not gt_map:
        return 0.0
    correct_count = 0
    for item in predictions:
        if item['idx'] in gt_map:
            # Clean both the prediction and ground truth for a stricter comparison
            cleaned_pred = clean_text(item['pred'])
            cleaned_gt = clean_text(gt_map[item['idx']])
            if cleaned_pred == cleaned_gt:
                correct_count += 1
    return correct_count / len(gt_map) if gt_map else 0.0


def evaluate_miou(predictions, gt_map):
    """Calculates Mean Intersection over Union over the entire ground truth set."""
    if not gt_map:
        return 0.0
        
    total_iou = 0.0
    # Create a set of predicted indices for efficient lookup
    predicted_indices = {item['idx'] for item in predictions}

    # Iterate through predictions to sum the IoU for valid entries
    for item in predictions:
        if item['idx'] not in gt_map:
            continue  # Ignore predictions that don't have a ground truth counterpart

        pred_val = item['pred']
        gt_val = gt_map[item['idx']] # No .get() needed since we know it's in the map

        # Standardize inputs to lists/sets
        pred_list = pred_val if isinstance(pred_val, list) else [pred_val]
        gt_list = gt_val if isinstance(gt_val, list) else [gt_val]
        
        pred_set = set(pred_list)
        gt_set = set(gt_list)

        intersection = len(pred_set.intersection(gt_set))
        union = len(pred_set.union(gt_set))
        
        total_iou += (intersection / union) if union > 0 else 0.0

    return total_iou / len(gt_map)

def evaluate_miou_trimmed(predictions, gt_map):
    """Calculates Mean Intersection over Union over the entire ground truth set."""
    if not gt_map:
        return 0.0
        
    total_iou = 0.0
    # Create a set of predicted indices for efficient lookup
    predicted_indices = {item['idx'] for item in predictions}

    counter_gt = 0
    for item in gt_map:
        gt_val = gt_map[item] 
        gt_list = gt_val if isinstance(gt_val, list) else [gt_val]
        if len(gt_list)==10:
            continue
        else:
            counter_gt +=1

    # Iterate through predictions to sum the IoU for valid entries
    for item in predictions:
        if item['idx'] not in gt_map:
            continue  # Ignore predictions that don't have a ground truth counterpart

        pred_val = item['pred']
        gt_val = gt_map[item['idx']] # No .get() needed since we know it's in the map
 
        # Standardize inputs to lists/sets
        pred_list = pred_val if isinstance(pred_val, list) else [pred_val]
        gt_list = gt_val if isinstance(gt_val, list) else [gt_val]

        if len(gt_list)==10:
            continue
        
        pred_set = set(pred_list)
        gt_set = set(gt_list)

        intersection = len(pred_set.intersection(gt_set))
        union = len(pred_set.union(gt_set))
        
        total_iou += (intersection / union) if union > 0 else 0.0

    return total_iou / counter_gt

def print_results_table(results, column_order):
    """
    Prints a formatted table summarizing the evaluation results.

    Args:
        results (dict): A dictionary of the form {model_name: {q_type: score, ...}}.
        column_order (list): A list of metric keys defining the column order.
    """
    if not results:
        print("No results to display.")
        return

    print("\n" + "="*80)
    print(" " * 25 + "MODEL PERFORMANCE SUMMARY")
    print("="*80)

    # 1. Use the provided column_order for the table columns
    all_q_types = column_order
    
    # 2. Define column widths for alignment
    model_col_width = max(len(model) for model in results.keys()) + 2 if results else 10
    q_type_col_width = 15

    # 3. Print header
    header = f"{'Model':<{model_col_width}}"
    for q_type in all_q_types:
        # Modify specific headers for display purposes
        display_q_type = q_type
        if q_type == "temporal_action_exact":
            display_q_type = "action_exact"
        elif q_type == "temporal_action_mc":
            display_q_type = "action_mc"
        
        header += f"| {display_q_type[:q_type_col_width-1]:<{q_type_col_width}} "
    print(header)
    print("-" * len(header))

    # 4. Print each model's results
    for model_name, model_scores in sorted(results.items()):
        row = f"{model_name:<{model_col_width}}"
        for q_type in all_q_types:
            score = model_scores.get(q_type) # Use .get() to handle missing scores
            score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
            row += f"| {score_str:<{q_type_col_width}} "
        print(row)
    
    print("="*80)

def main(pred_dir, gt_dir, trimmed_iou, eval_mode):
    """
    Main function to run the evaluation. Discovers model prediction folders and
    evaluates each one against the ground truth files.
    """


    PREDS_DIR = pred_dir


    #GT_DIR = gt_dir
    TAD = load_from_disk(gt_dir)

    #PREDS_DIR = "predictions"
    METRIC_DISPATCHER = {
        'exact_answer_action': evaluate_exact_match, 
        'mc_action': evaluate_accuracy,
        'action_duration': evaluate_accuracy,
        'temp_ordering': evaluate_accuracy,
        "temp_action_localize": evaluate_miou,
        "relative_action_localize": evaluate_accuracy,
        "temp_object_localize": evaluate_miou,
    }

    if trimmed_iou:
        METRIC_DISPATCHER['action_to_keysegments'] = evaluate_miou_trimmed

    print(f"--- Starting Evaluation (Mode: {eval_mode.upper()}) ---")
    
    # This dictionary will store all results for the final table
    all_results = defaultdict(dict)

    try:
        model_folders = [d for d in os.listdir(PREDS_DIR) if os.path.isdir(os.path.join(PREDS_DIR, d))]
    except FileNotFoundError:
        print(f"Error: Predictions directory not found at '{PREDS_DIR}'. Aborting.")
        return

    if not model_folders:
        print("No model folders found in the 'predictions' directory. Nothing to evaluate.")
        return

    # Loop over each model's sub-directory
    for model_name in sorted(model_folders):
        print(f"\n--- Evaluating Model: {model_name} ---")
        model_pred_dir = os.path.join(PREDS_DIR, model_name)
        pred_files = [f for f in os.listdir(model_pred_dir) if f.endswith('.jsonl')]
        
        if not pred_files:
            print(f"No prediction files found for model '{model_name}'. Skipping.")
            continue

        # Loop over the prediction files for the current model
        for pred_filename in pred_files:
            question_type = os.path.splitext(pred_filename)[0]
            pred_filepath = os.path.join(model_pred_dir, pred_filename)

            print(f"\n  Found Prediction for Question Type: '{question_type}'")
            print(f"    - Pred File:    {pred_filepath}")


            user_predictions = load_jsonl_file(pred_filepath)
            #ground_truth_data = load_jsonl_file(gt_filepath)
            ground_truth_data = TAD[question_type]

            if not user_predictions or not ground_truth_data:
                print("    -> Skipping evaluation due to one or both files being empty or unloadable.")
                continue

            if eval_mode != 'all':
                original_gt_count = len(ground_truth_data)
                                
                mixed_types = {
                    "exact_answer_action", "mc_action", 
                    "temp_ordering", "temp_action_localize"
                }
                ego_only_types = {"action_duration", "relative_action_localize"}
                non_ego_only_types = {"temp_object_localize"}
                ego_keyword = "the ego vehicle"
                
                filtered_gt = []

                if eval_mode == 'ego':
                    if question_type in ego_only_types:
                        filtered_gt = ground_truth_data
                    elif question_type in mixed_types:
                        filtered_gt = [
                            item for item in ground_truth_data 
                            if ego_keyword in item.get('question', '').lower()
                        ]
                elif eval_mode == 'non-ego':
                    if question_type in non_ego_only_types:
                        filtered_gt = ground_truth_data
                    elif question_type in mixed_types:
                        filtered_gt = [
                            item for item in ground_truth_data 
                            if ego_keyword not in item.get('question', '').lower()
                        ]
                
                ground_truth_data = filtered_gt

                if not ground_truth_data:
                    print(f"    -> No '{eval_mode}' questions found for '{question_type}'. Skipping.")
                    continue
                
                valid_indices = {item['idx'] for item in ground_truth_data}
                user_predictions = [pred for pred in user_predictions if pred['idx'] in valid_indices]
                
                print(f"    -> Filtered for '{eval_mode}' mode. Kept {len(ground_truth_data)} of {original_gt_count} questions.")

            gt_answer_map = {item['idx']: item['answer'] for item in ground_truth_data}
            eval_function = METRIC_DISPATCHER.get(question_type)

            if not eval_function:
                print(f"    -> Error: No metric defined for '{question_type}'. Skipping.")
                continue
                
            score = eval_function(user_predictions, gt_answer_map)
            metric_name = "Mean IoU"
            if eval_function == evaluate_accuracy:
                metric_name = "Accuracy"
            elif eval_function == evaluate_exact_match:
                metric_name = "Exact Match Accuracy"

            print(f"    -> Result: {metric_name}: {score:.4f}")
            
            # Store the result for the final table
            all_results[model_name][question_type] = score

    # --- Print the final summary table ---
    print_results_table(all_results, list(METRIC_DISPATCHER.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set predictions directory")
    parser.add_argument('--preds_dir', type=str, default="predictions", help="Directory to store predictions")
    parser.add_argument("--benchmark-files", type=str, default="/home/ma-user/work/saeed/TAD_code_data_submission/TAD/TAD_HF", help="Path to benchmark files")
    parser.add_argument("--trimmed", action="store_true",  help="Use evaluate_miou_trimmed instead of evaluate_miou for action_to_keysegments")
    parser.add_argument('--eval_mode', type=str, default='all', choices=['all', 'ego', 'non-ego'], help="Evaluation mode: 'all' for combined, 'ego' for ego-vehicle only, 'non-ego' for non-ego only.")
    args = parser.parse_args()
    main(args.preds_dir, args.benchmark_files, args.trimmed, args.eval_mode)