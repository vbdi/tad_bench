import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

import numpy as np
from pyquaternion import Quaternion

# lmdeploy for model inference
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

# nuscenes-devkit for loading data
from nuscenes.nuscenes import NuScenes
import argparse
import os

from datasets import load_dataset


STATIONARY_THRESHOLD_MS = 0.2         # m/s. Vehicle is considered stopped if most frames are below this.
STOPPING_SPEED_THRESHOLD = 1.0        # m/s. Used to distinguish 'starting' from 'stopping'.
TURN_YAW_THRESHOLD_DEG = 10.0         # Total yaw change in degrees over the sequence to be a turn.
LANE_CHANGE_LATERAL_VEL_THRESHOLD = 0.4 # m/s. Average lateral speed to be a lane change.
MIN_LANE_CHANGE_FORWARD_SPEED = 1.0   # m/s. Must be moving forward for a lane change.


def initialize_model(model_path):
    """Initializes and returns the InternVL model pipeline."""
    print(f"Initializing model: {model_path}...")
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

def create_scene_segment_map(keysegments_data):
    """
    Creates a nested dictionary for efficient lookup: scene_token -> segment_key_token -> [frame_tokens].
    Uses the 'keyframes_token_formatted' field created during processing.
    """
    scene_map = {}
    print("Pre-processing keysegments data for efficient lookup...")
    # Iterate over the dataset
    for item in keysegments_data:
        scene_token = item['scene_token']
        segments_for_scene = {}
        
        # Structure is list of dicts: [{'segment_key': '...', 'frames': [...]}, ...]
        formatted_segments = item.get('keyframes_token_formatted', [])
        
        for segment_item in formatted_segments:
            s_key = segment_item['segment_key']
            frames = segment_item['frames']
            segments_for_scene[s_key] = frames
                
        scene_map[scene_token] = segments_for_scene
        
    return scene_map

def get_image_paths_for_scene(nusc, sample_tokens):
    """Gets the CAM_FRONT image paths for a list of sample_tokens."""
    image_paths = []
    for token in sample_tokens:
        sample_record = nusc.get('sample', token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        image_path = nusc.get_sample_data_path(cam_front_token)
        image_paths.append(image_path)
    return image_paths

def get_raw_ego_pose_details(nusc: NuScenes, frame_tokens: List[str]) -> str:
    """
    Generates a textual summary of the ego-vehicle's raw pose data for a list of frames.
    This is used when the RAW_FOR_BASELINE flag is True.
    """
    if not frame_tokens:
        return "No ego-vehicle pose data available."

    pose_details = []
    for i, sample_token in enumerate(frame_tokens):
        sample = nusc.get('sample', sample_token)
        # Get the ego pose token from the CAM_FRONT sample_data
        pose_token = nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']
        ego_pose = nusc.get('ego_pose', pose_token)

        translation = ego_pose['translation']
        # pyquaternion's __str__ method provides the desired format (e.g., w + xi + yj + zk)
        rotation = Quaternion(ego_pose['rotation'])
        timestamp = ego_pose['timestamp']

        # Format the string as per the user's request
        detail_str = (
            f"The ego pose details for Frame{i+1} are "
            f"translation={translation}, "
            f"rotation={str(rotation)}, "
            f"timestamp={timestamp}"
        )
        pose_details.append(detail_str)

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


def format_prompt(question_item, num_images, motion_summary: str = None, blind=False, thinking=False):
    """
    Formats the prompt with image placeholders, optional motion summary/data, question, and options.
    """
    instruction = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " if thinking else ""

    if blind:
        #context_str = ""
        image_str = "A forward-facing video from a car driving through urban streets, showing roads, vehicles, pedestrians, and city infrastructure is given."
        motion_str = motion_summary if motion_summary else ""
        context_str = f"{image_str}\n{motion_str}".strip()
    else:
        image_placeholders = [f'Frame{i+1}: {IMAGE_TOKEN}' for i in range(num_images)]
        image_str = '\n'.join(image_placeholders)

        # Add motion summary or raw data if provided
        motion_str = motion_summary if motion_summary else ""

        # Combine image placeholders and motion context
        context_str = f"{image_str}\n{motion_str}".strip()

    question_text = question_item['question']

    prompt_parts = [context_str, question_text]

    if 'options' in question_item and question_item['options']:
        options_text = '\n'.join(question_item['options'])
        prompt_parts.append(options_text)

    if thinking:
        prompt_parts.append(instruction)

    # Join the parts, filtering out any empty strings
    return '\n'.join(filter(None, prompt_parts))


def parse_thinking_output(response_text: str) -> (str, str):
    """
    Parses the model's response to extract the thinking process and the final answer.
    """
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)

    thinking_process = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else response_text

    return thinking_process, answer_text

def extract_mc_answer(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.)
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
    Parses the model's response to extract the single-letter answer (A, B, C, etc.).
    This is used for multiple-choice questions, including cases where the answer
    is inside \boxed{}.
    """
    # Create a list of possible starting letters like ['A', 'B', 'C', ...]
    option_letters = [opt[0] for opt in options if opt and opt[0].isalpha() and opt[1] == '.']

    # 1. Check for \boxed{X}
    boxed_match = re.search(r'\\boxed\{([A-Z])\}', response_text)
    if boxed_match:
        letter = boxed_match.group(1)
        if letter in option_letters:
            return letter

    # 2. Check for patterns like "A.", "B ", "Answer: A", etc.
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

def main(args):
    NUSCENES_VERSION = args.nuscenes_version
    NUSCENES_DATAROOT = args.nuscenes_dataroot 
    BENCHMARK_FILES = args.benchmark_files
    QUESTIONS_FILES = args.questions_files 
    OUTPUT_DIR = args.output_dir
    # --- Model Configuration ---
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    BLIND_INFERENCE = args.blind_inference
    THINKING = args.thinking
    MAX_PATCH  = args.max_patch
    RAW_FOR_BASELINE = args.raw_for_baseline     
    
    pipe = initialize_model(MODEL_PATH)
    nusc = initialize_nuscenes(NUSCENES_DATAROOT, NUSCENES_VERSION)

    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')

    scene_segment_map = create_scene_segment_map(keysegments_data)

    for questions_file in QUESTIONS_FILES:
        print(f"\n--- Processing Question File: {questions_file} ---")

        questions_data = load_dataset(os.path.join(BENCHMARK_FILES, questions_file), split='train')


        model_output_dir = OUTPUT_DIR / MODEL_NAME
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / f"{questions_file}.jsonl"

        print(f"Predictions will be saved to: {output_file}")

        with open(output_file, 'w') as f:
            pass

        for question in tqdm(questions_data, desc=f"Answering {questions_file}"):
            scene_token = question['scene_token']
            segment_key_token = question['sample_token']
            idx = question['idx']

            frame_tokens = scene_segment_map.get(scene_token, {}).get(segment_key_token)

            if not frame_tokens:
                print(f"Warning: Segment '{segment_key_token}' for scene '{scene_token}' not found. Skipping question idx {idx}.")
                continue

            ego_motion_context = None
            if frame_tokens: # Check if the list is not empty
                if RAW_FOR_BASELINE:
                    # Generate raw pose details for all frames
                    ego_motion_context = get_raw_ego_pose_details(nusc, frame_tokens)
                else:
                    # Generate motion summary only if there are at least two frames
                    if len(frame_tokens) >= 2:
                        start_token, end_token = frame_tokens[0], frame_tokens[-1]
                        summary = get_motion_summary_from_poses(nusc, start_token, end_token)
                        ego_motion_context = f"Motion summary: {summary}"


            try:
                image_paths = get_image_paths_for_scene(nusc, frame_tokens)

                prompt = format_prompt(
                    question,
                    len(image_paths),
                    motion_summary=ego_motion_context,
                    blind=BLIND_INFERENCE,
                    thinking=THINKING
                )

                content = []
                content.append(dict(type='text', text=prompt))
                min_pixels = args.min_pixels #256 * 28 * 28
                max_pixels = args.max_pixels #512 * 28 * 28

                if not BLIND_INFERENCE:
                    for image_path in image_paths:
                        content.append(
                            dict(
                                type='image_url',
                                image_url=dict(
                                    max_dynamic_patch=MAX_PATCH,
                                    min_pixels=min_pixels,
                                    max_pixels=max_pixels,
                                    url=image_path)))

                messages = [dict(role='user', content=content)]
                gen_config = GenerationConfig(max_new_tokens=1024)

                response = pipe(messages, gen_config=gen_config)

                raw_response_text = response.text
                thinking_process, answer_to_parse = parse_thinking_output(raw_response_text)

                if 'options' in question and question['options']:
                    if BLIND_INFERENCE:
                        prediction = extract_answer_blind(answer_to_parse, question['options'])
                    else:
                        prediction = extract_mc_answer(answer_to_parse, question['options'])
                else:
                    prediction = answer_to_parse.strip()

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

    print("\n--- All question files processed successfully! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NuScenes inference configuration")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="1",
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
        "--questions-files",
        type=str,  
        nargs="+",
        default=[
            "exact_answer_action",
            "mc_action"
        ],
        help="Task names "
    )

    parser.add_argument(
        "--benchmark-files",
        type=str,  
        default="/home/ma-user/work/saeed/TAD_code_data_submission/TAD/TAD_HF",
        help="Path to benchmark files"
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

    args = parser.parse_args()
    args.nuscenes_dataroot = Path(args.nuscenes_dataroot)
    args.output_dir = Path(args.output_dir)
   # Apply GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("### Running on GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    args.model_name = f"{Path(args.model_path).name}_{args.exp_name}"
    # Print configuration summary
    print("Benchmark data path:", args.benchmark_files)
    print("Model path:", args.model_path)
    print("Model name:", args.model_name) 
    main(args)