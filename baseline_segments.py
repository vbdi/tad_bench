import json
import re
from pathlib import Path
from tqdm import tqdm

# lmdeploy for model inference
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

# nuscenes-devkit for loading data
from nuscenes.nuscenes import NuScenes
import argparse
import os

from datasets import load_dataset

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

def format_prompt(question_item, num_images, blind=False, thinking=False):
    """
    Formats the prompt with image placeholders, question, and options (if available).
    Adds a special instruction if thinking is enabled.
    """
    instruction = "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " if thinking else ""

    if blind:
        image_str = "A forward-facing video from a car driving through urban streets, showing roads, vehicles, pedestrians, and city infrastructure is given."
    else:

        image_placeholders = [f'Frame{i+1}: {IMAGE_TOKEN}' for i in range(num_images)]
        image_str = '\n'.join(image_placeholders)
    
    question_text = question_item['question']
    
    # Check if the question has options (i.e., it's a multiple-choice question)
    if 'options' in question_item and question_item['options']:
        options_text = '\n'.join(question_item['options'])
        if thinking:
            prompt = f"{image_str}\n{question_text}\n{options_text}\n{instruction}"
        else:
            prompt = f"{image_str}\n{question_text}\n{options_text}"

    else:
        # If no options, it's an exact-answer question. The prompt is just the question.
        if thinking:
            prompt = f"{image_str}\n{question_text}\n{instruction}"
        else:
            prompt = f"{image_str}\n{question_text}"
        
    return prompt.strip()

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

def extract_mc_answer(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.)
    for multiple-choice questions.
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

def extract_mc_answer_blind(response_text: str, options: list) -> str:
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


def main(args):    
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

    # Initialization
    pipe = initialize_model(MODEL_PATH)
    nusc = initialize_nuscenes(NUSCENES_DATAROOT, NUSCENES_VERSION)

    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')

    # Load and pre-process the keysegments file into an efficient map.
    scene_segment_map = create_scene_segment_map(keysegments_data)

    
    # --- Main loop ---
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

        # --- Loop through each question in the current file ---
        for question in tqdm(questions_data, desc=f"Answering {questions_file}"):
            scene_token = question['scene_token']
            segment_key_token = question['sample_token'] 
            idx = question['idx']
            
            # --- Perform data lookup for the current question ---
            frame_tokens = scene_segment_map.get(scene_token, {}).get(segment_key_token)
            
            if not frame_tokens:
                print(f"Warning: Segment '{segment_key_token}' for scene '{scene_token}' not found. Skipping question idx {idx}.")
                continue
                
            try:
                image_paths = get_image_paths_for_scene(nusc, frame_tokens)
                
                prompt = format_prompt(question, len(image_paths), blind=BLIND_INFERENCE, thinking=THINKING)
                                
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

                # Check if the question is multiple-choice
                if 'options' in question and question['options']:
                    if BLIND_INFERENCE:
                        prediction = extract_mc_answer_blind(answer_to_parse, question['options'])
                    else:    
                        prediction = extract_mc_answer(answer_to_parse, question['options'])
                else:
                    # For exact-answer questions, prediction is the cleaned answer text
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