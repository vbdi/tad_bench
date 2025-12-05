from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from tqdm import tqdm                           
from cot_utils import *                          
import os    
import argparse                                   

from datasets import load_dataset

def main(args):    

    # parameter initialization
    QUESTIONS_FILES = args.questions_files 
    BENCHMARK_FILES = args.benchmark_files
    OUTPUT_DIR = args.output_dir
    CAPTION_START_IDX = 34                  # Used to remove a "description of video segment" that is added during captioning, which could be repetitive/confusing.

    # --- Model Configuration ---
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    ENABLE_THINKING = args.thinking
    DO_COT = args.use_cot_captions
    LLM_PROMPT_OPTION = args.llm_prompt_option
    INPUT_CAPTION_FILE = Path(args.captions_path)

    # Initialization
    llm_model, llm_tokenizer = initialize_llm(MODEL_PATH)

    # HF load
    keysegments_data = load_dataset(os.path.join(BENCHMARK_FILES, 'keysegments'), split='train')
    scene_segment_map = create_scene_segment_map(keysegments_data)
    scene_data_map = create_scene_data_map(keysegments_data)
    
    # --- Main loop ---
    for questions_file in QUESTIONS_FILES:
        print(f"\n--- Processing Question File: {questions_file} ---")
        
        # HF load
        questions_data = load_dataset(os.path.join(BENCHMARK_FILES, questions_file), split='train')
        
        # Prepare the model-specific output directory and file path
        model_output_dir = OUTPUT_DIR / args.model_name                                     
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / f"{questions_file}.jsonl"
        
        print(f"Predictions will be saved to: {output_file}")
        
        # Clear the output file before writing new results
        with open(output_file, 'w') as f:
            pass  

        # CoT captions are loaded for all videos togehter from a single JSON
        if(DO_COT == True):
            captions_all_videos = load_data(INPUT_CAPTION_FILE)
            scene_lut = make_scene_lut(captions_all_videos)

        # --- Loop through each question in the current file ---
        for question in tqdm(questions_data, desc=f"Answering {questions_file}"):
            try:
                scene_token = question['scene_token']               # scene token
                idx = question['idx']                               # question number
                segment_key_token = question['sample_token']        # token for the segment in this question
                scene_data = scene_data_map.get(scene_token)        # scene info
                
                if not scene_data:
                    print(f"Warning: Scene token {scene_token} from questions file not found in keysegments file. Skipping.")
                    continue

                # For CoT captions, extract the specific caption for this scene
                if(DO_COT == True):
                    all_scene_info = scene_lut[scene_token]
                    all_captions = []
                    seg_count = 1                    

                    for c_seg_info in all_scene_info:

                        c_segment_description = "The following information provides a description of video keysegment " + str(seg_count) + ": \n"

                        # traffic scene description
                        c_segment_description += "Scene Description: \n"                        
                        c_segment_description += c_seg_info.get("chain_of_thought_history", {}).get("step1_scene_description", "None\n")

                        # JSON-style summary
                        c_segment_description += "\nSummary Description: "
                        c_segment_description += c_seg_info["caption"][CAPTION_START_IDX:]           

                        all_captions.extend([c_segment_description])  
                        seg_count += 1  
                # For non-CoT captions, load the JSON file for this scene
                else:
                    with open(INPUT_CAPTION_FILE / (str(scene_token) + ".json"), 'r') as f:
                        json_data =  json.load(f)
                        all_captions = json_data["caption_outputs"] 

                # Use the pre-processed map for a fast, direct lookup
                frame_tokens = scene_segment_map.get(scene_token, {}).get(segment_key_token)
                keysegments_lookup = scene_data.get('keysegments')
                seg_caption_index = keysegments_lookup.index(question["keysegment"])

                if not frame_tokens:
                    print(f"Warning: Segment '{segment_key_token}' for scene '{scene_token}' not found in keysegments file. Skipping question idx {idx}.")
                    continue

                # use the question and the the single caption for the specific segment to construct LLM's prompt
                qa_prompt = format_prompt_cap2ans(question, [all_captions[seg_caption_index]],LLM_PROMPT_OPTION)

                # set up the LLM
                if("Qwen3-14B" in MODEL_NAME):
                    messages = [                        
                        {"role": "user", "content": qa_prompt}
                    ]                         
                    text = llm_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=ENABLE_THINKING
                    )
                    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

                    generated_ids = llm_model.generate(
                        **model_inputs,
                        max_new_tokens=16384,
                        do_sample=False                        
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    qa_response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                elif("Qwen2.5-14B-Instruct-1M" in MODEL_NAME or "Qwen2.5-7B-Instruct" in MODEL_NAME):
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": qa_prompt}
                    ]                    
                    text = llm_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

                    generated_ids = llm_model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        do_sample=False
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    qa_response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Check if the question is multiple-choice
                if 'options' in question and question['options']:
                    prediction = extract_answer(qa_response, question['options'])
                else:
                    answer_prefix = "<answer>"
                    answer_suffix = "</answer>"

                    if answer_prefix in qa_response and answer_suffix in qa_response:
                        ans_start_idx = qa_response.find(answer_prefix) + len(answer_prefix)
                        ans_end_idx = qa_response.find(answer_suffix)
                        real_ans_text = qa_response[ans_start_idx:ans_end_idx]
                        qa_response = real_ans_text

                    prediction = qa_response.strip()
                
                # Prepare result and append to file immediately
                result = {"idx": idx, "prompt": qa_prompt, "pred": prediction, "raw_output" : qa_response}
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
        default="3",
        help="Comma-separated list of GPU device IDs to make visible to CUDA"
    )
    parser.add_argument(
        "--keysegments-file",
        type=str,  
        default="TAD/generated_questions/keysegments.jsonl",
        help="Path to keysegments JSONL file"
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
        default="kc_predictions_2",
        help="Directory to save predictions"
    )
    # --- Model Configuration ---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ma-user/work/pretrained_models/models--Qwen--Qwen2.5-14B-Instruct-1M/snapshots/620fad32de7bdd2293b3d99b39eba2fe63e97438/",
        help="Path to the LLM for answering the questions."
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Qwen2.5-14B-Instruct-1M",
        help="Name of the LLM. ***NOTE: This should correspond to the path.***"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="Scene-CoT",
        help="Experiment name to append to model path name"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode"
    )
    parser.add_argument(
        "--use_cot_captions",
        action="store_true",
        help="Must turn on this flag if using CoT captions."
    )
    parser.add_argument(
        "--llm_prompt_option",
        type=int,
        default=3,
        help="Select the specific LLM prompt."
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default="cot_captions/InternVL3-8B_captions.jsonl",
        help="Path to captions. CoT captions are stored in single jsonl"
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

   # Apply GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("### Running on GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    caption_model_name = os.path.basename(args.captions_path).split("_")[0] 
    args.model_name = f"CaptionModel={caption_model_name}_LLM={args.llm_name}_CoTCaps={args.use_cot_captions}_{args.exp_name}"
    # Print configuration summary
    print("Benchmark data path:", args.benchmark_files)
    print("Model path:", args.model_path)
    print("Model name:", args.model_name)
    main(args)