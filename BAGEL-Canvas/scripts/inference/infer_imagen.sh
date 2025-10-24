#!/bin/bash

# Automatically calculate the number of GPUs to use
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Starting Imagen Inference for BAGEL-Canvas"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES:-All}"
echo "Number of GPUs: $NUM_GPUS"
echo "========================================"

# ==============================================================================
# ||                           CONFIGURATION                                  ||
# ||           MODIFY YOUR PATHS, MODEL, AND INFERENCE PARAMETERS HERE.       ||
# ==============================================================================

# --- 1. Path and Model Configuration ---
# Path to the base model configuration (contains config.json, tokenizer, etc.)
MODEL_PATH="your_model_path/BAGEL-Canvas"
# Path to the model checkpoint (contains ema.safetensors or model.safetensors)
CKPT_PATH="your_model_path/BAGEL-Canvas"
# Root directory where the final generated images will be saved
OUTPUT_ROOT="your_save_path/mathvanvas_imagen_inference"

# --- 2. Dataset Configuration ---
# dataset name, choices: code_derived_captions, repurposing_competition, repurposing_foundational, mavis, tr_cot
DATASET_NAME="code_derived_captions"
# Path to the Parquet file containing 'id', 'caption', and 'image' columns
PARQUET_FILE="your_data_path/MathCanvas-Imagen/data/${DATASET_NAME}/val/val-00000-of-00001.parquet"

# --- 3. Inference Parameters ---
EVAL_EMA="model"              # Which model weights to use: "ema" or "model"
RESOLUTION=512              # Resolution of the generated images
NUM_IMAGES_PER_PROMPT=4     # Number of images to generate per prompt
TOTAL_PROMPTS_TO_RUN=100    # Total number of prompts to sample from the dataset for testing
TIMESTEP_SHIFT=3.0          # Timestep shift for the diffusion process
CFG_SCALE=4.0               # CFG-Scale for guidance
NUM_TIMESTEPS=50            # Number of diffusion steps
ENABLE_TAYLORSEER="false"   # Whether to enable Taylorseer acceleration: "true" or "false"
SEED=42                     # Random seed for reproducibility

# ==============================================================================
# ||                EXECUTION LOGIC                                           ||
# ==============================================================================

# Automatically adjust batch size based on resolution
if [[ $RESOLUTION -ge 1024 ]]; then
    BATCH_SIZE=1
else
    BATCH_SIZE=4
fi
echo "Using Batch Size: $BATCH_SIZE"

# Build the Taylorseer argument
TAYLORSEER_ARG=""
if [[ "${ENABLE_TAYLORSEER,,}" == "true" ]]; then
    echo "Taylorseer is enabled."
    TAYLORSEER_ARG="--enable_taylorseer"
else
    echo "Taylorseer is disabled."
fi

# Print configuration for confirmation
echo "----------------------------------------"
echo "Model Path:      $MODEL_PATH"
echo "Checkpoint Path: $CKPT_PATH"
echo "Parquet File:    $PARQUET_FILE"
echo "Output Dir:      $OUTPUT_DIR"
echo "----------------------------------------"

# Execute the torchrun command
torchrun \
    --nnodes=1 --node_rank=0 --nproc_per_node=$NUM_GPUS --master_addr=localhost --master_port=12345 \
    -m mathcanvas_imgen_inferencer \
    --model-path "$MODEL_PATH" \
    --ckpt-path "$CKPT_PATH" \
    --eval_ema "$EVAL_EMA" \
    --dataset_name "$DATASET_NAME" \
    --parquet_file "$PARQUET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --gen_num "$TOTAL_PROMPTS_TO_RUN" \
    --num_images "$NUM_IMAGES_PER_PROMPT" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --resolution "$RESOLUTION" \
    --timestep_shift "$TIMESTEP_SHIFT" \
    --cfg_scale "$CFG_SCALE" \
    --num_timesteps "$NUM_TIMESTEPS" \
    $TAYLORSEER_ARG

echo "========================================"
echo "Inference for dataset '$DATASET_NAME' completed."
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"