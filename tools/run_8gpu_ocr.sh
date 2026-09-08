#!/usr/bin/env bash
# Edit the values in this block when starting the next 8-GPU shard.
INPUT_DIR="/home/jovyan/myh-data-ceph-shcdt-3/dingkunyuan/data/arxiv_cn/prepared"
OUTPUT_ROOT="/home/jovyan/wxy-ceph-sh/dingkunyuan/data/ocr_annotations_8gpu"
MODEL_DIR="/home/jovyan/myh-data-ceph-shcdt-3/dingkunyuan/code/EasyOCR/.ocr_models"
FIRST_SAMPLE=1
SAMPLES_PER_GPU=10000
BATCH_SIZE=16
VALIDATION_SAMPLES=2
CPU_THREADS=4

mkdir -p "$OUTPUT_ROOT/logs"

for GPU in {0..7}; do
  START_SAMPLE=$((FIRST_SAMPLE + GPU * SAMPLES_PER_GPU))
  END_SAMPLE=$((START_SAMPLE + SAMPLES_PER_GPU - 1))
  PART_DIR="$OUTPUT_ROOT/part$((GPU + 1))"
  LOG_FILE="$OUTPUT_ROOT/logs/gpu${GPU}.log"

  nohup python -u tools/build_ocr_annotations.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$PART_DIR" \
    --model_storage_directory "$MODEL_DIR" \
    --download_disabled \
    --gpu "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --cpu_threads "$CPU_THREADS" \
    --start_sample "$START_SAMPLE" \
    --end_sample "$END_SAMPLE" \
    --validation_samples "$VALIDATION_SAMPLES" \
    --resume > "$LOG_FILE" 2>&1 &
  echo "GPU $GPU: samples $START_SAMPLE-$END_SAMPLE, pid=$!, log=$LOG_FILE"
done
