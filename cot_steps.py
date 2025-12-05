import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# lmdeploy for model inference
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

# nuscenes-devkit for loading data
from nuscenes.nuscenes import NuScenes
import argparse

from datasets import load_dataset


NUM_SAMPLED_FRAMES = 4 

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


def sample_frames(frame_tokens: list, num_frames: int) -> list:
    """
    Uniformly samples a specified number of frames from a list of frame tokens.
    If the list has fewer frames than required, it returns all of them.
    """
    if len(frame_tokens) <= num_frames:
        return frame_tokens
    
    indices = np.linspace(0, len(frame_tokens) - 1, num_frames, dtype=int)
    sampled_tokens = [frame_tokens[i] for i in indices]
    return sampled_tokens

def get_image_paths_for_scene(nusc, sample_tokens):
    """Gets the CAM_FRONT image paths for a list of sample_tokens."""
    image_paths = []
    for token in sample_tokens:
        sample_record = nusc.get('sample', token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        image_path = nusc.get_sample_data_path(cam_front_token)
        image_paths.append(image_path)
    return image_paths

# --- Chain-of-Thought Prompting Functions ---

def format_step1_scene_description_prompt(num_images: int) -> str:
    """Formats the prompt for Step 1: General Scene Description."""
    image_placeholders = [f'Frame{i+1}: {IMAGE_TOKEN}' for i in range(num_images)]
    image_str = '\n'.join(image_placeholders)

    prompt = """You are an expert in autonomous driving scene understanding. You are given a sequence of frames from the front camera of a vehicle.
Your first task is to provide a concise, high-level description of the overall scene including the nearby vehicles with distinguishable descriptions. 
Describe the scene in a few sentences. Focus only on what you can see."""
    
    full_prompt = f"{image_str}\n\n{prompt}"
    return full_prompt.strip()

def format_step2_ego_motion_prompt(step1_output: str) -> str:
    """Formats the prompt for Step 2: Ego Vehicle Motion Analysis."""
    prompt = f"""Based on the provided frames and your previous scene description:
---
{step1_output}
---
Now, focus *only* on the ego vehicle's motion. Analyze how the background and lane markings move across the frames. Is the ego vehicle moving forward, turning, changing lanes, starting from a stop, or stopping?

Describe your reasoning in one or two sentences, and then conclude with the most likely motion type.
**Your final conclusion must be one of these exact phrases**:
- Stopped
- Turn left
- Turn right
- Change lane to the left
- Change lane to the right
- Starting
- Stopping
- Straight, constant speed"""
    return prompt.strip()

def format_step3_nearby_vehicles_prompt(step1_output, step2_output: str) -> str:
    """Formats the prompt for Step 3: Nearby Vehicles Motion Analysis."""
    prompt = f"""Excellent. Your analysis of the scene and the ego vehicle's motion was:
---
{step1_output}
{step2_output}
---
Now, analyze nearby vehicles (cars, trucks, buses, bicycles, motorcycles, trailer, construction vehicle) in front of or near the ego vehicle. For each distinct vehicle, provide a brief reasoning and conclude with its most likely motion type (including stopped) from the following list:
- Stopped
- Turn left
- Turn right
- Change lane to the left
- Change lane to the right
- Starting
- Stopping
- Straight, constant speed

If there are no other  vehicles or their motion is unclear, state that."""
    return prompt.strip()

def format_step4_final_json_prompt(step2_output: str, step3_output: str) -> str:
    """Formats the prompt for Step 4: Final JSON Generation."""
    classification_to_phrase = {
        "Stopped": "stopped.",
        "Turn left": "turning to the left.",
        "Turn right": "turning to the right.",
        "Change lane to the left": "changing lane to the left.",
        "Change lane to the right": "changing lane to the right.",
        "Starting": "starting from a stop.",
        "Stopping": "stopping.",
        "Straight, constant speed": "moving forward at a relatively constant speed."
    }

    prompt = f"""Based on all the previous analysis:
---
Ego Vehicle Analysis Summary:
{step2_output}
---
Nearby Vehicles Analysis Summary:
{step3_output}
---
Your final task is to consolidate this information into a single JSON object. 
- First, extract the final motion classification for the ego vehicle and each nearby vehicle from your previous responses.
- Second, map each classification to the corresponding descriptive phrase using the provided dictionary.
- Finally, construct the JSON object.

**Motion Phrases Dictionary**:
{json.dumps(classification_to_phrase, indent=2)}

**Output Format**:
{{
  "ego_vehicle_motion": "<phrase from dictionary>",
  "nearby_vehicles_motion": [
    {{
      "vehicle_id": "<a brief, unique description, e.g., 'white SUV in front'>",
      "motion": "<phrase from dictionary>"
    }},
    ...
  ]
}}

**Important**: Respond with *only* the raw JSON object and nothing else. Do not wrap it in markdown or add any explanations."""
    return prompt.strip()

def insert_description(caption_text, seg_num):
    description_line = f"Description of video keysegment {seg_num}:"
    
    # Split the string into a list of lines
    lines = caption_text.splitlines()
    
    # Find the index of the opening "```json" line
    try:
        json_start_index = lines.index("```json")
    except ValueError:
        raise ValueError("No JSON block found in caption text.")
    
    # Insert the description line before the JSON block
    lines.insert(json_start_index, description_line)
    
    # Join the list back into a single string
    return "\n".join(lines)

def ensure_json_block(s: str) -> str:
    """
    Ensures the given string starts with a ```json code block indicator.
    If it's missing, it will be added.
    """
    s = s.strip()
    if not s.startswith("```json"):
        # Add the json code block markers
        s = f"```json\n{s}\n```"
    return s

def main(args):
    NUSCENES_VERSION = args.nuscenes_version
    NUSCENES_DATAROOT = args.nuscenes_dataroot 
    BENCHMARK_FILES = args.benchmark_files
    OUTPUT_DIR = args.output_dir
    # --- Model Configuration ---
    MODEL_PATH = args.model_path
    MAX_PATCH  = args.max_patch    

    # Initialization
    pipe = initialize_model(MODEL_PATH)
    nusc = initialize_nuscenes(NUSCENES_DATAROOT, NUSCENES_VERSION)

    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')
    scene_segment_map = create_scene_segment_map(keysegments_data)
    
    model_name_tag = Path(MODEL_PATH).name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{model_name_tag}_captions.jsonl"
    print(f"Temporal captions will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        pass  

    tasks = []
    for scene_token, segments in scene_segment_map.items():
        for segment_key_token, frame_tokens in segments.items():
            tasks.append((scene_token, segment_key_token, frame_tokens))

    previous_scene_token = "previous token"
    segmet_counter = 1 
    # --- Main processing loop with Chain-of-Thought ---
    for scene_token, segment_key_token, all_frame_tokens in tqdm(tasks, desc="Generating Temporal Captions"):
        if previous_scene_token in scene_token:
            segmet_counter += 1
        else:
            segmet_counter = 1
        try:
            sampled_frame_tokens = sample_frames(all_frame_tokens, num_frames=NUM_SAMPLED_FRAMES)
            
            if not sampled_frame_tokens:
                print(f"Warning: No frames to process for segment '{segment_key_token}'. Skipping.")
                continue

            image_paths = get_image_paths_for_scene(nusc, sampled_frame_tokens)
            
            messages = []
            gen_config = GenerationConfig(max_new_tokens=1024, temperature=0.0)

            prompt1 = format_step1_scene_description_prompt(len(image_paths))
            content1 = [dict(type='text', text=prompt1)]
            min_pixels = args.min_pixels #256 * 28 * 28
            max_pixels = args.max_pixels #512 * 28 * 28
            for image_path in image_paths:
                content1.append(dict(
                    type='image_url',
                    image_url=dict(
                        max_dynamic_patch=MAX_PATCH,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        url=image_path
                    )
                ))
            messages.append(dict(role='user', content=content1))
            
            response1 = pipe(messages, gen_config=gen_config)
            step1_output = response1.text.strip()
            messages.append(dict(role='assistant', content=step1_output))

            prompt2 = format_step2_ego_motion_prompt(step1_output)
            messages.append(dict(role='user', content=prompt2))

            response2 = pipe(messages, gen_config=gen_config)
            step2_output = response2.text.strip()
            messages.append(dict(role='assistant', content=step2_output))

            # --- STEP 3: Nearby Vehicles Motion ---
            prompt3 = format_step3_nearby_vehicles_prompt(step1_output, step2_output)
            messages.append(dict(role='user', content=prompt3))

            response3 = pipe(messages, gen_config=gen_config)
            step3_output = response3.text.strip()
            messages.append(dict(role='assistant', content=step3_output))

            prompt4 = format_step4_final_json_prompt(step2_output, step3_output)
            messages.append(dict(role='user', content=prompt4))

            final_response = pipe(messages, gen_config=gen_config)
            final_json_output = final_response.text.strip()
            #print(final_json_output)
            final_json_output = ensure_json_block(final_json_output)
            #print(final_json_output)
            final_json_output = insert_description(final_json_output,segmet_counter)

            # 6. Assemble and append the final result to the output file
            result = {
                "scene_token": scene_token,
                "segment_token": segment_key_token,
                "caption": final_json_output,
                "chain_of_thought_history": {
                    "step1_scene_description": step1_output,
                    "step2_ego_motion_analysis": step2_output,
                    "step3_nearby_vehicles_analysis": step3_output,
                }
            }
            previous_scene_token = scene_token

            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nAn error occurred while processing segment {segment_key_token} for scene {scene_token}: {e}")
            error_result = {
                "scene_token": scene_token,
                "segment_token": segment_key_token,
                "caption": f"ERROR: {str(e)}"
            }

            previous_scene_token = scene_token

            with open(output_file, 'a') as f:
                f.write(json.dumps(error_result) + '\n')

    print("\n--- Temporal captioning complete! ---")
    print(f"All captions have been saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NuScenes inference configuration")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="5",
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
        "--output-dir",
        type=str,  
        default="scene_cot_captions/",
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
    # Print configuration summary
    print("Model path:", args.model_path)
    main(args)