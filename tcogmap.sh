#!bin/bash
MODEL_PATH=/home/ma-user/work/pretrained_models/InternVL3-8B/
NuScenes_PATH=/data/AD/NuScenes
BENCHMARK_PATH=vbdai___tad/


OUTPUT_PATH=predictions
EXP_NAME="TCogMap"

# Extract the model name from the MODEL_PATH (last directory name)
MODEL_NAME=$(basename "$MODEL_PATH")

# Define log directory
LOG_DIR="${OUTPUT_PATH}/${MODEL_NAME}_${EXP_NAME}/log"
mkdir -p "$LOG_DIR"

#===========================================
# Motion summaries
#===========================================
python tcogmap_segments.py \
    --gpu-id 2 \
    --nuscenes-dataroot "$NuScenes_PATH" \
    --benchmark-files "$BENCHMARK_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --exp-name "$EXP_NAME" \
    > "${LOG_DIR}/tcogmap_segments.log" 2>&1 &


python tcogmap_scenes.py \
    --gpu-id 3 \
    --nuscenes-dataroot "$NuScenes_PATH" \
    --benchmark-files "$BENCHMARK_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --exp-name "$EXP_NAME" \
    --full-frames \
    > "${LOG_DIR}/tcogmap_scenes.log" 2>&1 &

# Wait for both to finish
wait

echo "All processes finished. Logs are in: $LOG_DIR"

python ./evaluation.py --preds_dir $OUTPUT_PATH