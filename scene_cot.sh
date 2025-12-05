#!bin/bash
#==========================
# VLM part for reasoning 
#=========================
MODEL_PATH=/home/ma-user/work/pretrained_models/InternVL3-8B/
NuScenes_PATH=data/AD/NuScenes
BENCHMARK_PATH=vbdai___tad/
OUTPUT_PATH=scene_cot_captions_hf/

# Extract the model name from the MODEL_PATH (last directory name)
MODEL_NAME=$(basename "$MODEL_PATH")

# Define log directory
LOG_DIR="${OUTPUT_PATH}/${MODEL_NAME}/log"
mkdir -p "$LOG_DIR"

python ./cot_steps.py \
    --gpu-id 4 \
    --nuscenes-dataroot "$NuScenes_PATH" \
    --benchmark-files "$BENCHMARK_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_PATH" \
    > "${LOG_DIR}/cot_steps.log" 2>&1

# =======================================================
# LLM Infer Part 
# =====================================================
# captions path
CAPTIONS_PATH="${OUTPUT_PATH}/InternVL3-8B_captions.jsonl"


MODEL_PATH=/home/ma-user/work/pretrained_models/models--Qwen--Qwen2.5-14B-Instruct-1M/snapshots/620fad32de7bdd2293b3d99b39eba2fe63e97438/
MODEL_NAME=Qwen2.5-14B-Instruct-1M
OUTPUT_PATH=predictions

python cot_scenes.py \
--gpu-id 5 \
--captions_path $CAPTIONS_PATH \
--benchmark-files "$BENCHMARK_PATH" \
--model-path $MODEL_PATH \
--llm_name $MODEL_NAME \
--output-dir $OUTPUT_PATH \
--exp-name "Scene_CoT_HF_Check" \
--use_cot_captions &

python cot_segments.py \
--gpu-id 6 \
--captions_path $CAPTIONS_PATH \
--benchmark-files "$BENCHMARK_PATH" \
--model-path $MODEL_PATH \
--llm_name $MODEL_NAME \
--output-dir $OUTPUT_PATH \
--exp-name "Scene_CoT_HF_Check" \
--use_cot_captions &

wait
python evaluation.py --preds_dir $OUTPUT_PATH