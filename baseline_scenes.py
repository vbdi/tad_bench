import json
import re
from pathlib import Path
from tqdm import tqdm

import os

# lmdeploy for model inference
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

# nuscenes-devkit for loading data
from nuscenes.nuscenes import NuScenes
from datasets import load_dataset

import argparse

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

def get_all_sample_tokens_for_scene(nusc, scene_token):
    """
    Retrieves all sample tokens for a given scene by traversing the linked list
    of samples from the first to the last.
    """
    all_sample_tokens = []
    scene_record = nusc.get('scene', scene_token)
    current_sample_token = scene_record['first_sample_token']
    
    # Iterate through the samples until the 'next' token is empty
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

def format_prompt(question_item, num_images, blind=False, thinking=False):
    """
    Formats the prompt with image placeholders and question.
    Includes options only if they exist in the question item.
    Adds a special instruction if thinking is enabled.
    """
    
    instruction = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " if thinking else ""

    if blind:
        image_str = "A forward-facing video from a car driving through urban streets, showing roads, vehicles, pedestrians, and city infrastructure is given."
    else:
        image_placeholders = [f'Frame{i+1}: {IMAGE_TOKEN}' for i in range(num_images)]
        image_str = '\n'.join(image_placeholders)
    
    # Format question
    question_text = question_item['question']
    
    # Conditionally add options
    if 'options' in question_item and question_item['options']:
        options_text = '\n'.join(question_item['options'])
        if thinking:
            prompt = f"{image_str}\n{question_text}\n{options_text}\n{instruction}"
        else:
            prompt = f"{image_str}\n{question_text}\n{options_text}"
    else:
        # For questions without multiple-choice options (like the new task)
        if thinking:
            prompt = f"{image_str}\n{question_text}\n{instruction}"
        else:
            prompt = f"{image_str}\n{question_text}"
        
    return prompt.strip()

def extract_answer(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.).
    This is used for multiple-choice questions.
    """
    # Create a list of possible starting letters like ['A', 'B']
    option_letters = [opt[0] for opt in options if opt and opt[0].isalpha() and opt[1] == '.']

    # Search for patterns like "A.", "B. ", "Answer: A", etc.
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

def parse_thinking_output(response_text: str) -> (str, str):
    """
    Parses the model's response to extract the thinking process and the final answer.
    
    Args:
        response_text: The raw text output from the model.
        
    Returns:
        A tuple containing:
        - thinking_process (str): The text within <think> tags, or None if not found.
        - answer_text (str): The text within <answer> tags. If tags are not found,
                             this defaults to the original response_text.
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


            for keyframe_group in keyframes_index_data:
                keysegment_str = keyframe_group.get('segment_index_key')                 
                associated_frames = keyframe_group.get('indices', [])
                
                if not associated_frames:  # Avoid division by zero or empty processing
                    continue

                # Count 
                num_predicted_in_group = sum(1 for frame_idx in associated_frames if frame_idx in predicted_frames_set)

                if num_predicted_in_group>0: # num_predicted_in_group > len(associated_frames) / 2: >  if majority
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


    # Initialization
    pipe = initialize_model(MODEL_PATH)
    nusc = initialize_nuscenes(NUSCENES_DATAROOT, NUSCENES_VERSION)
    
    
    print("Loading common data files...")
    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')

    scene_data_map = create_scene_data_map(keysegments_data)

    # Loop over each questions file
    for questions_file in QUESTIONS_FILES:
        print(f"\n--- Processing Question File: {questions_file} ---")
        
        # HF load
        questions_data = load_dataset(os.path.join(BENCHMARK_FILES, questions_file), split='train')
        
        model_output_dir = OUTPUT_DIR / MODEL_NAME
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = model_output_dir / f"{questions_file}.jsonl"
        
        print(f"Predictions will be saved to: {output_file}")
        
        with open(output_file, 'w') as f:
            pass  # Clear the file initially

        # Main inference loop for the current question file
        for question in tqdm(questions_data, desc=f"Processing {questions_file}"):
            scene_token = question['scene_token']
            idx = question['idx']
            
            scene_data = scene_data_map.get(scene_token)
            if not scene_data:
                print(f"Warning: Scene token {scene_token} from questions file not found in keysegments file. Skipping.")
                continue
            
            # Now, determine which set of frames to use.
            if FULL_FRAMES:
                sample_tokens = get_all_sample_tokens_for_scene(nusc, scene_token)
            else:
                # Use the pre-sampled frames from the keysegments file.
                sample_tokens = scene_data['sample_tokens']

            try:
                image_paths = get_image_paths_for_scene(nusc, sample_tokens)
                prompt = format_prompt(question, len(image_paths), blind=BLIND_INFERENCE, thinking=THINKING)
                
                content = []
                content.append(dict(type='text', text=prompt))
                
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
                    if scene_data:
                        prediction = parse_and_map_keysegments(answer_to_parse, scene_data, FULL_FRAMES)
                    else:
                        print(f"Warning: No scene data found for scene {scene_token}. Prediction will be empty.")
                        prediction = []
                else:
                    # Original task: Extract single letter answer
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
        default="baseline",
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
    # Print configuration summary
    print("Benchmark data path:", args.benchmark_files)
    print("Model path:", args.model_path)
    print("Model name:", args.model_name)
    args.full_frames = True
    print("Full Frame flags:", args.full_frames) 
    main(args)