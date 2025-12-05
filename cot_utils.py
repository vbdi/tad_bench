from transformers import AutoModelForCausalLM, AutoTokenizer        
import json
import re


def initialize_llm(llm_path):
    """ Initializes the LLM."""
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype="auto",
        device_map="auto"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)

    return llm_model, llm_tokenizer
  
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

def format_prompt_cap2ans(question_item, all_captions, LLM_PROMPT_OPTION):
    """
    Formats the prompt with image placeholders, question, and options (if available).
    Adds a special instruction if thinking is enabled.
    """
    question_text = question_item['question']

    if(LLM_PROMPT_OPTION == 3):
        caption_text = ("Use the following descriptions of the video keysegments to assist in answering the question. Each keysegment corresponds to a "
                        "short, five-second snippet of the original video. The original video is of a traffic scene from the perspective of an autonomous driving "
                        "vehicle. In addition to using the video keysegment captions themselves, use logic, reasoning, and knowledge of traffic rules "
                        "to answer the question. You may provide some justification, but please enclose the succint, final answer to the question between <answer></answer> tags.\n")


    for c_caption in all_captions:
        caption_text += c_caption + "\n\n"

    # Check if the question has options (i.e., it's a multiple-choice question)
    if 'options' in question_item and question_item['options']:
        options_text = '\n'.join(question_item['options'])
        prompt = f"\n{question_text}\n{options_text}\n{caption_text}"

    else:
        # If no options, it's an exact-answer question. The prompt is just the question.
        prompt = f"{question_text}\n{caption_text}"
    
    return prompt.strip()


def extract_answer(response_text: str, options: list) -> str:
    """
    Parses the model's response to extract the single-letter answer (A, B, C, etc.).
    This is used for multiple-choice questions.
    """
    # Create a list of possible starting letters like ['A', 'B']
    option_letters = [opt[0] for opt in options if opt and opt[0].isalpha() and opt[1] == '.']

    # Consider answers that may be contained in special output tags.
    answer_prefix = "<answer>"
    answer_suffix = "</answer>"
    boxed_prefix = "\\boxed{"
    boxed_suffix = "}"

    if answer_prefix in response_text and answer_suffix in response_text:
        ans_start_idx = response_text.find(answer_prefix) + len(answer_prefix)
        ans_end_idx = response_text.find(answer_suffix)
        real_ans_text = response_text[ans_start_idx:ans_end_idx]
        match = re.search(r'([A-Z])', real_ans_text)                # Assmues that if the model has used this type of tag, the first and only letter will be the answer

    elif boxed_prefix in response_text:
        ans_start_idx = response_text.find(boxed_prefix) + len(boxed_prefix)
        ans_end_idx = response_text.find(boxed_suffix)
        real_ans_text = response_text[ans_start_idx:ans_end_idx]
        match = re.search(r'([A-Z])', real_ans_text)                 # Assmues that if the model has used this type of tag, the first and only letter will be the answer
    else:
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

def parse_and_map_keysegments(response_text: str, scene_data_item: dict, full_frames: bool = False) -> list:
    """
    Parses the model's response to extract a list of indices.
    If full_frames is True, it maps the predicted frame indices back to keysegments
    If full_frames is False, it maps 1-based keysegment indices to actual
    keysegment values using the 'keysegments' list.
    """
    # Find a list-like structure in the text, e.g., "[1, 2, 3]" or " [1,2, 3] "
    text = response_text.strip()
    
    # Consider answers that may be contained in special output tags.
    answer_prefix = "<answer>"
    answer_suffix = "</answer>"

    # if the answer tag exists, grab what is in between the answer tags
    if answer_prefix in text and answer_suffix in text:
        ans_start_idx = text.find(answer_prefix) + len(answer_prefix)
        ans_end_idx = text.find(answer_suffix)
        real_ans_text = text[ans_start_idx:ans_end_idx]
        text = real_ans_text
  
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
        
def make_scene_lut(captions_all_videos):
    """
    Creates a an efficient scene_token -> Complete caption lookup table.
    """
    lookup_dict = {}

    for c_caption in captions_all_videos:
        if(lookup_dict.get(c_caption['scene_token'], None) == None):
            lookup_dict[c_caption['scene_token']] = [c_caption]
        else:
            lookup_dict[c_caption['scene_token']].extend([c_caption])

    return lookup_dict   


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


def create_scene_segment_map_indices(keysegments_data):
    """
    Creates a nested dictionary for efficient lookup: scene_token -> key_segment_index -> [key_frame_indices].
    This pre-processes the 'keyframes_token' data for fast access.
    """
    scene_map = {}
    print("Pre-processing keysegments data for efficient lookup via indices...")
    for item in keysegments_data:
        scene_token = item['scene_token']
        # Create a sub-dictionary for the segments of the current scene
        segments_for_scene = {}
        
        # The 'keyframes_index' field contains the list of segment indices
        keyframe_segments = item.get('keyframes_index', [])
        
        for segment_dict in keyframe_segments:
            # Each dict has one key (the segment identifier) and one value (the list of frame indices)
            # .items() is used to safely extract the key and value
            for segment_key, frame_indices in segment_dict.items():
                segments_for_scene[segment_key] = frame_indices
                
        scene_map[scene_token] = segments_for_scene
    return scene_map