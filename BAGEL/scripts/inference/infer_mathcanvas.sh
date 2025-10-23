#!/bin/bash

# Automatically calculate the number of GPUs to use
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Starting interleave reasoning Inference for BAGEL-Canvas on MathCanvas Benchmark"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES:-All}"
echo "Number of GPUs: $NUM_GPUS"
echo "========================================"

torchrun --nproc_per_node=$NUM_GPUS --master_port=23456 \
    mathcanvas_interleave_reasoner.py \
    --dataset_type mathcanvas \
    --dataset_name mathcanvas \
    --input_path ../benchmark/data.jsonl \
    --checkpoint_dir your_model_path/BAGEL-Canvas \
    --checkpoint_file model.safetensors \
    --model_path your_model_path/BAGEL-Canvas \
    --image_root ../benchmark \
    --output_dir your_save_path/BAGEL-Canvas \
    --add_timestamp false \
    --run_description default \
    --max_iterations 10 \
    --skip_completed \
    --do_sample true \
    --text_temperature 0.3 \
    --cfg_text_scale 4.0 \
    --cfg_img_scale 2.0 \
    --cfg_interval 0.0 1.0 \
    --timestep_shift 3.0 \
    --num_timesteps 50 \
    --cfg_renorm_min 0.0 \
    --cfg_renorm_type text_channel \
    --enable_taylorseer false \