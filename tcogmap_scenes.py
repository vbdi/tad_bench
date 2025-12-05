import json
import re
from pathlib import Path
from tqdm import tqdm
import ast
import os
from typing import List, Dict, Any

# --- New imports for motion summary generation ---
import numpy as np
from pyquaternion import Quaternion

# lmdeploy for model inference
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

from nuscenes.nuscenes import NuScenes
import argparse

from datasets import load_dataset

STATIONARY_THRESHOLD_MS = 0.2         # m/s. Vehicle is considered stopped if most frames are below this.
STOPPING_SPEED_THRESHOLD = 1.0        # m/s. Used to distinguish 'starting' from 'stopping'.
TURN_YAW_THRESHOLD_DEG = 10.0         # Total yaw change in degrees over the sequence to be a turn.
LANE_CHANGE_LATERAL_VEL_THRESHOLD = 0.4 # m/s. Average lateral speed to be a lane change.
MIN_LANE_CHANGE_FORWARD_SPEED = 1.0   # m/s. Must be moving forward for a lane change.

def initialize_model(model_path):
    """Initializes and returns the InternVL model pipeline."""
    print(f"Initializing model: {model_path}...")
    # Use TurbomindEngineConfig for better performance
    backend_config = TurbomindEngineConfig(session_len=32768, tp=1)
    
    pipe = pipeline(
        model_path,
        backend_config=backend_config
    )
    print("Model initialized successfully.")
    return pipe

def initialize_nuscenes(nuscenes_dataroot, nuscenes_version):
    """Initializes and returns the NuScenes dataset object."""
    print(f"Initializing NuScenes (version: {nuscenes_version})...")
    if not nuscenes_dataroot.exists():
        raise FileNotFoundError(
            f"NuScenes dataroot not found at: {nuscenes_dataroot}\n"
            "Please update the NUSCENES_DATAROOT variable in the script."
        )
    nusc = NuScenes(version=nuscenes_version, dataroot=str(nuscenes_dataroot), verbose=False)
    print("NuScenes initialized successfully.")
    return nusc

def load_data(file_path):
    """Loads a .jsonl file into a list of dictionaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def create_scene_data_map(keysegments_data):
    """
    Creates a dictionary mapping scene_token to its full data object
    from the keysegments file.
    """
    scene_map = {}
    for item in keysegments_data:
        scene_map[item['scene_token']] = item
    return scene_map

# Helper function to get all frames for a scene
def get_all_sample_tokens_for_scene(nusc, scene_token):
    """
    Retrieves all sample tokens for a given scene by traversing the linked list
    of samples from the first to the last.
    """
    all_sample_tokens = []
    scene_record = nusc.get('scene', scene_token)
    current_sample_token = scene_record['first_sample_token']
    
    while current_sample_token:
        all_sample_tokens.append(current_sample_token)
        sample_record = nusc.get('sample', current_sample_token)
        current_sample_token = sample_record['next']
        
    return all_sample_tokens

def get_image_paths_for_scene(nusc, sample_tokens):
    """Gets the CAM_FRONT image paths for a list of sample_tokens."""
    image_paths = []
    for token in sample_tokens:
        sample_record = nusc.get('sample', token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        image_path = nusc.get_sample_data_path(cam_front_token)
        image_paths.append(image_path)
    return image_paths

def get_raw_ego_pose_details(nusc: NuScenes, start_sample_token: str, end_sample_token: str, start_frame_idx: int) -> str:
    """
    Generates a textual block of raw ego-vehicle pose data for each frame between two samples.
    """
    pose_details = []
    current_token = start_sample_token
    frame_counter = start_frame_idx
    
    while True:
        sample = nusc.get('sample', current_token)
        pose_token = nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']
        ego_pose = nusc.get('ego_pose', pose_token)
        
        # Format the pose details for the current frame
        translation = ego_pose['translation']
        # Use pyquaternion's string representation for clear, consistent output
        rotation = Quaternion(ego_pose['rotation']) 
        timestamp = ego_pose['timestamp']
        
        detail_str = (
            f"Frame{frame_counter + 1}: translation={translation}, "
            f"rotation={rotation}, timestamp={timestamp}"
        )
        pose_details.append(detail_str)
        
        if current_token == end_sample_token or not sample['next']:
            break
            
        current_token = sample['next']
        frame_counter += 1
        
    return "\n".join(pose_details)


def _classify_motion_from_data_points(data_points: List[Dict[str, Any]]) -> str:
    """
    Analyzes a sequence of structured data points (translation, rotation, velocity, timestamp)
    and classifies the motion. This is the core logic, reusable for any vehicle.
    Returns a classification string like "Stopped", "Turn left", etc.
    """
    motion_classification = "Undefined"

    if len(data_points) < 2:
        speed = np.linalg.norm(data_points[0]['velocity'][:2]) if data_points else 0
        if speed < STATIONARY_THRESHOLD_MS:
            motion_classification = "Stopped"
    else:
        start_point = data_points[0]
        end_point = data_points[-1]
        
        speeds_ms = [np.linalg.norm(p['velocity'][:2]) for p in data_points]
        
        if sum(s < STATIONARY_THRESHOLD_MS for s in speeds_ms) > len(speeds_ms) / 2:
            motion_classification = "Stopped"
        else:
            start_speed = speeds_ms[1] 
            end_speed = speeds_ms[-1]

            start_yaw_rad = start_point['rotation'].yaw_pitch_roll[0]
            end_yaw_rad = end_point['rotation'].yaw_pitch_roll[0]
            yaw_change_rad = end_yaw_rad - start_yaw_rad
            yaw_change_rad = (yaw_change_rad + np.pi) % (2 * np.pi) - np.pi
            yaw_change_deg = np.rad2deg(yaw_change_rad)

            local_velocities = [p['rotation'].inverse.rotate(p['velocity']) for p in data_points]
            avg_local_vx = np.mean([v[0] for v in local_velocities])
            avg_local_vy = np.mean([v[1] for v in local_velocities])

            if abs(yaw_change_deg) > TURN_YAW_THRESHOLD_DEG:
                motion_classification = "Turn left" if yaw_change_deg > 0 else "Turn right"
            elif abs(avg_local_vy) > LANE_CHANGE_LATERAL_VEL_THRESHOLD and avg_local_vx > MIN_LANE_CHANGE_FORWARD_SPEED:
                motion_classification = "Change lane to the left" if avg_local_vy > 0 else "Change lane to the right"
            elif start_speed < STOPPING_SPEED_THRESHOLD and end_speed > STOPPING_SPEED_THRESHOLD * 1.5:
                motion_classification = "Starting"
            elif start_speed > STOPPING_SPEED_THRESHOLD * 1.5 and end_speed < STOPPING_SPEED_THRESHOLD:
                motion_classification = "Stopping"
            else:
                motion_classification = "Straight, constant speed"

    return motion_classification

def get_motion_summary_from_poses(nusc: NuScenes, start_sample_token: str, end_sample_token: str) -> str:
    """Generates a textual summary of the ego-vehicle's motion between two samples."""
    ego_poses_raw = []
    current_token = start_sample_token
    while True:
        sample = nusc.get('sample', current_token)
        pose_token = nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']
        ego_poses_raw.append(nusc.get('ego_pose', pose_token))
        
        if current_token == end_sample_token or not sample['next']:
            break
        current_token = sample['next']
    
    if not ego_poses_raw:
        return "The ego-vehicle's motion could not be determined."

    data_points: List[Dict[str, Any]] = []
    for i, current_pose in enumerate(ego_poses_raw):
        point = {
            'translation': np.array(current_pose['translation']),
            'rotation': Quaternion(current_pose['rotation']),
            'timestamp': current_pose['timestamp']
        }
        
        if i == 0:
            point['velocity'] = np.array([0.0, 0.0, 0.0])
        else:
            prev_point = data_points[i-1]
            time_diff_s = (point['timestamp'] - prev_point['timestamp']) / 1e6
            velocity = (point['translation'] - prev_point['translation']) / time_diff_s if time_diff_s > 1e-6 else np.array([0.0, 0.0, 0.0])
            point['velocity'] = velocity
        
        data_points.append(point)

    motion_classification = _classify_motion_from_data_points(data_points)

    prefix = "The ego-vehicle is"
    classification_to_phrase = {
        "Stopped": "stopped.", 
        "Turn left": "turning to the left.",
        "Turn right": "turning to the right.",
        "Change lane to the left": "changing lane to the left.",
        "Change lane to the right": "changing lane to the right.",
        "Starting": "starting from a stop.",
        "Stopping": "stopping.",
        "Straight, constant speed": "moving forward at a relatively constant speed.",
    }
    phrase = classification_to_phrase.get(motion_classification, "in an undefined state.")
    return f"{prefix} {phrase}"

def format_prompt(question_item, num_images, motion_summaries: List[str] = None, blind=False, thinking=False):
    """
    Formats the prompt with image placeholders, optional motion summaries, and question.
    """
    instruction = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " if thinking else ""

    if blind:
        image_str = "A forward-facing video from a car driving through urban streets, showing roads, vehicles, pedestrians, and city infrastructure is given."
        motion_str = '\n'.join(motion_summaries) if motion_summaries else ""
    else:
        # Part 1: Image placeholders
        image_placeholders = [f'Frame{i+1}: {IMAGE_TOKEN}' for i in range(num_images)]
        image_str = '\n'.join(image_placeholders)

        # Part 2: Motion summaries (if provided)
        motion_str = '\n'.join(motion_summaries) if motion_summaries else ""

    # Combine parts
    context_str = f"{image_str}\n{motion_str}".strip()

    # Format question
    question_text = question_item['question']
    
    prompt = ""
    # Conditionally add options
    if 'options' in question_item and question_item['options']:
        options_text = '\n'.join(question_item['options'])
        prompt = f"{context_str}\n{question_text}\n{options_text}"
    else:
        # For questions without multiple-choice options (like the new task)
        prompt = f"{context_str}\n{question_text}"
    
    if thinking:
        prompt = f"{prompt}\n{instruction}"
        
    return prompt.strip()


def extract_answer(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.).
    """
    option_letters = [opt[0] for opt in options if opt and opt[0].isalpha() and opt[1] == '.']

    match = re.search(r'([A-Z])[\.\s\:\)]', response_text)
    if match:
        letter = match.group(1)
        if letter in option_letters:
            return letter
            
    cleaned_text = response_text.strip()
    if len(cleaned_text) == 1 and cleaned_text in option_letters:
        return cleaned_text
        
    return "N/A"

def extract_answer_blind(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.)
    for multiple-choice questions, including cases where the answer is inside \boxed{}.
    """
    # Collect valid option letters (A, B, C, ...)
    option_letters = [opt[0] for opt in options if opt and opt[0].isalpha() and opt[1] == '.']

    # 1. Check for \boxed{X}
    boxed_match = re.search(r'\\boxed\{([A-Z])\}', response_text)
    if boxed_match:
        letter = boxed_match.group(1)
        if letter in option_letters:
            return letter

    # 2. Check for patterns like "Answer: B" or "Correct answer is B."
    match = re.search(r'([A-Z])[\.\s:\)]', response_text)
    if match:
        letter = match.group(1)
        if letter in option_letters:
            return letter

    # 3. If the entire response is just a single letter
    cleaned_text = response_text.strip()
    if len(cleaned_text) == 1 and cleaned_text in option_letters:
        return cleaned_text

    return "N/A"


def parse_thinking_output(response_text: str) -> (str, str):
    """
    Parses the model's response to extract the thinking process and the final answer.
    """
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    
    thinking_process = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else response_text
    
    return thinking_process, answer_text

def parse_and_map_keysegments(response_text: str, scene_data_item: dict, full_frames: bool = False) -> list:
    """
    Parses the model's response to extract a list of indices.
    If full_frames is True, it maps the predicted frame indices back to keysegments
    If full_frames is False, it maps 1-based keysegment indices to actual
    keysegment values using the 'keysegments' list.
    """
    # Find a list-like structure in the text, e.g., "[1, 2, 3]" or " [1,2, 3] "
    text = response_text.strip()
    
    candidates = []
    PRIORITY = {
        "lines":   0,
        "match_4": 1,
        "match_1": 2,
        "match_2": 3,
        "match_3": 4,
    }

    # 1. Bracketed list, allowing optional quotes around it
    match_1 = re.search(r'\[?\s*\d+(?:[\s,]+\d+)*\s*\]?', text)

    # 2. After "keysegment(s)" or "frame(s)"
    match_2 = re.search(r'(?:keysegments?|frames?)\s+([\d,\sand]+)', text, re.IGNORECASE)

    # 3. Bulleted/dashed lines with "keysegment(s)" or "frame(s)"
    match_3 = re.findall(r'[-•]?\s*(?:keysegments?|frames?)\s*(\d+)', text, re.IGNORECASE)

    # 4. Standalone numbers on separate lines
    lines = [line.strip() for line in text.splitlines() if line.strip().isdigit()]

    # 5. Plain comma-separated numbers (allow trailing punctuation/quotes)
    match_4 = re.fullmatch(r'\s*\d+(?:[\s,]+\d+)*\s*[.,;:]?\s*', re.sub(r'[\'"]$', '', text))


    if match_1:
        candidates.append(("match_1",list(map(int, re.findall(r'\d+', match_1.group())))))
    if match_2:
        candidates.append(("match_2",list(map(int, re.findall(r'\d+', match_2.group(1))))))
    if match_3:
        candidates.append(("match_3",list(map(int, match_3))))
    if lines:
        candidates.append(("lines",list(map(int, lines))))
    if match_4:
        candidates.append(("match_4",list(map(int, re.findall(r'\d+', match_4.group())))))
    
    if candidates:
        predicted_indices = max(
            candidates,
            key=lambda x: (len(x[1]), -PRIORITY[x[0]])  # longest first, then highest priority
        )[1]
    else:
        print(f"Warning: Could not find a valid list in the model response: '{response_text}'")
        return []

    try:
        if full_frames:
            # Map predicted frame indices back to keysegments via majority vote.
            mapped_values = []
            keyframes_index_data = scene_data_item.get('keyframes_index_formatted', [])
            if not keyframes_index_data:
                print(f"Warning: 'keyframes_index_formatted' not found for scene {scene_data_item.get('scene_token')}. Cannot map predictions.")
                return []

            # Use a set for efficient lookup of predicted frames
            predicted_frames_set = set(predicted_indices)

           # Iterate through each keysegment's associated frame list
            for keyframe_group in keyframes_index_data:
                
                keysegment_str = keyframe_group.get('segment_index_key')                 
                associated_frames = keyframe_group.get('indices', [])
                
                if not associated_frames:  # Avoid division by zero or empty processing
                    continue

                # Count 
                num_predicted_in_group = sum(1 for frame_idx in associated_frames if frame_idx in predicted_frames_set)

                if num_predicted_in_group>0: # num_predicted_in_group > len(associated_frames) / 2: for majority
                    try:
                        # The keysegment value is the key of the dict, convert to int
                        mapped_values.append(int(keysegment_str))
                    except ValueError:
                        print(f"Warning: Could not convert keysegment '{keysegment_str}' to an integer.")
            
            return sorted(mapped_values)  # Sort for consistent output
        else:
            # ORIGINAL LOGIC for when FULL_FRAMES is False
            keysegments_lookup = scene_data_item.get('keysegments', [])
            if not keysegments_lookup:
                 print(f"Warning: 'keysegments' not found for scene {scene_data_item.get('scene_token')}. Cannot map predictions.")
                 return []
            
            mapped_values = []
            for index in predicted_indices:
                # Model predicts a 1-based index into the keysegments list
                if 1 <= index <= len(keysegments_lookup):
                    # Model predicts 1 -> use lookup[0], predicts 2 -> use lookup[1], etc.
                    mapped_values.append(keysegments_lookup[index - 1])
                else:
                    print(f"Warning: Invalid index {index} predicted. It's outside the valid range [1, {len(keysegments_lookup)}]. Skipping.")
                    
            return mapped_values

    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse the model's prediction list. Error: {e}. Response: '{response_text}'")
        return []

def main(args): 
    # parameter initialization
    NUSCENES_VERSION = args.nuscenes_version
    NUSCENES_DATAROOT = args.nuscenes_dataroot 
    QUESTIONS_FILES = args.questions_files 
    BENCHMARK_FILES = args.benchmark_files
    OUTPUT_DIR = args.output_dir
    # --- Model Configuration ---
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    BLIND_INFERENCE = args.blind_inference
    THINKING = args.thinking
    MAX_PATCH  = args.max_patch
    FULL_FRAMES = args.full_frames
    RAW_FOR_BASELINE = args.raw_for_baseline     
 

    pipe = initialize_model(MODEL_PATH)
    nusc = initialize_nuscenes(NUSCENES_DATAROOT, NUSCENES_VERSION)
    
    print("Loading common data files...")
    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')
    scene_data_map = create_scene_data_map(keysegments_data)
    
    for questions_file in QUESTIONS_FILES:
        print(f"\n--- Processing Question File: {questions_file} ---")
        
        # HF load
        questions_data = load_dataset(os.path.join(BENCHMARK_FILES, questions_file), split='train')
        
        model_output_dir = OUTPUT_DIR / MODEL_NAME
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / f"{questions_file}.jsonl"
        
        print(f"Predictions will be saved to: {output_file}")
        
        with open(output_file, 'w') as f:
            pass

        for question in tqdm(questions_data, desc=f"Processing {questions_file}"):
            scene_token = question['scene_token']
            idx = question['idx']
            
            scene_data = scene_data_map.get(scene_token)
            if not scene_data:
                print(f"Warning: Scene token {scene_token} from questions file not found in keysegments file. Skipping.")
                continue
            
            try:
                if FULL_FRAMES:
                    sample_tokens = get_all_sample_tokens_for_scene(nusc, scene_token)
                    image_paths = get_image_paths_for_scene(nusc, sample_tokens)
                    
                    motion_summaries = []

                    keyframe_tokens_data = scene_data.get('keyframes_token_formatted', [])
                    keyframe_indices_data = scene_data.get('keyframes_index_formatted', [])

                    for token_item, index_item in zip(keyframe_tokens_data, keyframe_indices_data):
                        
                        # Extract the lists directly using the fixed keys from our transform function
                        token_list = token_item.get('frames', [])
                        index_list = index_item.get('indices', [])

                        # Basic validation
                        if len(token_list) < 2: 
                            continue
                        if not index_list: 
                            continue

                        start_token = token_list[0]
                        end_token = token_list[-1]
                        
                        start_frame_idx = index_list[0]
                        end_frame_idx = index_list[-1]
                        
                        if RAW_FOR_BASELINE:
                            ego_details_str = get_raw_ego_pose_details(nusc, start_token, end_token, start_frame_idx)
                            summary_text = (f"The ego pose details for Frame{start_frame_idx + 1} to Frame{end_frame_idx + 1} are:\n{ego_details_str}")
                        else:
                            ego_summary = get_motion_summary_from_poses(nusc, start_token, end_token)
                            summary_text = (f"Motion summary for Frame{start_frame_idx + 1} to Frame{end_frame_idx + 1}: {ego_summary}")
                        
                        motion_summaries.append(summary_text)

                prompt = format_prompt(
                    question, 
                    len(image_paths), 
                    motion_summaries=motion_summaries,
                    blind=BLIND_INFERENCE, 
                    thinking=THINKING
                )
                
                content = [dict(type='text', text=prompt)]
                min_pixels = args.min_pixels #256 * 28 * 28
                max_pixels = args.max_pixels #512 * 28 * 28
                
                if not BLIND_INFERENCE:
                    for image_path in image_paths:
                        content.append(dict(type='image_url', image_url=dict(max_dynamic_patch=MAX_PATCH, min_pixels=min_pixels, max_pixels=max_pixels, url=image_path)))
                                
                messages = [dict(role='user', content=content)]
                gen_config = GenerationConfig(max_new_tokens=1024)
                response = pipe(messages, gen_config=gen_config)

                raw_response_text = response.text
                thinking_process, answer_to_parse = parse_thinking_output(raw_response_text)

                is_keysegment_task = ('temp_action_localize' in questions_file) or ('temp_object_localize' in questions_file)

                if is_keysegment_task:
                    prediction = parse_and_map_keysegments(answer_to_parse, scene_data, FULL_FRAMES) if scene_data else []
                else:
                    if BLIND_INFERENCE:
                        prediction = extract_answer_blind(answer_to_parse, question['options'])
                    else:
                        prediction = extract_answer(answer_to_parse, question['options'])
                
                result = {"idx": idx, "prompt": prompt, "raw_response": raw_response_text, "pred": prediction}
                
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                    
            except Exception as e:
                print(f"An error occurred while processing question idx {idx} for scene {scene_token}: {e}")
                error_result = {"idx": idx, "prompt": "ERROR", "pred": "ERROR"}
                with open(output_file, 'a') as f:
                    f.write(json.dumps(error_result) + '\n')

        print(f"\nInference complete for {questions_file}!")
        print(f"All predictions have been saved to {output_file}")

    print("\n--- All question files processed! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NuScenes inference configuration")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="4",
        help="Comma-separated list of GPU device IDs to make visible to CUDA"
    )
    parser.add_argument(
        "--nuscenes-version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes dataset version"
    )
    parser.add_argument(
        "--nuscenes-dataroot",
        type=str,  
        default="/home/ma-user/work/kevin/data/AD/NuScenes",
        help="Path to NuScenes dataset root"
    )
    parser.add_argument(
        "--benchmark-files",
        type=str,  
        default="/home/ma-user/work/saeed/TAD_code_data_submission/TAD/TAD_HF",
        help="Path to benchmark files"
    )

    parser.add_argument(
        "--questions-files",
        type=str,  
        nargs="+",
        default=[
            'action_duration',
            'relative_action_localize',
            'temp_ordering',
            'temp_action_localize',
            'temp_object_localize',
        ],
        help="List of question JSONL files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,  
        default="predictions",
        help="Directory to save predictions"
    )

    # --- Model Configuration ---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ma-user/work/pretrained_models/InternVL3-8B/",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="tcogmap",
        help="Experiment name to append to model path name"
    )
    parser.add_argument(
        "--blind-inference",
        action="store_true",
        help="Enable blind inference mode"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode"
    )

    parser.add_argument(
        "--raw-for-baseline",
        action="store_true",
        help="Input raw ego poses into model context"
    )

    parser.add_argument(
        "--max-patch",
        type=int,
        default=1,
        help="Maximum patch size"
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum number of pixels"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Maximum number of pixels"
    )
    parser.add_argument(
        "--full-frames",
        action="store_true",
        help="Use all frames for inference"
    )

    args = parser.parse_args()
    # Convert relevant string args to Path objects after parsing
    args.nuscenes_dataroot = Path(args.nuscenes_dataroot)
    args.output_dir = Path(args.output_dir)
   # Apply GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("### Running on GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    args.model_name = f"{Path(args.model_path).name}_{args.exp_name}"
    args.full_frames = True
    # Print configuration summary
    print("Benchmark data path:", args.benchmark_files)
    print("Model path:", args.model_path)
    print("Model name:", args.model_name)
    print("Full Frame flags:", args.full_frames) 
    main(args)