#!bin/bash
#set these variables
MODEL_PATH=/home/ma-user/work/pretrained_models/InternVL3-8B/
NuScenes_PATH=/data/AD/NuScenes
BENCHMARK_PATH=vbdai___tad/

OUTPUT_PATH=predictions
EXP_NAME="baseline"

# Extract the model name from the MODEL_PATH (last directory name)
MODEL_NAME=$(basename "$MODEL_PATH")

# Define log directory
LOG_DIR="${OUTPUT_PATH}/${MODEL_NAME}_${EXP_NAME}/log"
mkdir -p "$LOG_DIR"

# #===========================================
# # Baselines
# #===========================================
python ./baseline\_segments.py \
  --gpu-id 0 \
  --nuscenes-dataroot "$NuScenes_PATH" \
  --benchmark-files "$BENCHMARK_PATH" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_PATH" \
  --exp-name "$EXP_NAME" \
  > "${LOG_DIR}/baseline_segments.log" 2>&1 &
  #> "${LOG_DIR}/inference_action.log" 2>&1
  

python ./baseline\_scenes.py \
  --gpu-id 1 \
  --nuscenes-dataroot "$NuScenes_PATH" \
  --benchmark-files "$BENCHMARK_PATH" \
  --model-path "$MODEL_PATH" \
  --exp-name "$EXP_NAME" \
  --output-dir "$OUTPUT_PATH" \
  --full-frames \
  > "${LOG_DIR}/baseline_scenes.log" 2>&1 &
  #> "${LOG_DIR}/inference_mc_and_action_2_segments.log" 2>&1 
  

# Wait for both to finish
wait

echo "Both processes finished. Logs are in: $LOG_DIR"

python ./evaluation.py --preds_dir $OUTPUT_PATH