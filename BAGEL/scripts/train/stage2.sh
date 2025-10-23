# Stage 2

MODEL_PATH="your_model_path/ByteDance-Seed/BAGEL-7B-MoT"
RESUME_FROM="your_save_path/mathcanvas_stage1--0/checkpoints/0080000"
RESUME_FROM=$RESUME_FROM


export WANDB_API_KEY="your_wandb_key"
WANDB_ENABLE="True"
WANDB_NAME="mathcanvas_stage2"
WANDB_RUNID="0"
WANDB_RESUME="allow"
WANDB_OFFLINE="False"

RESULTS_DIR="your_save_path/${WANDB_NAME}--${WANDB_RUNID}"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

torchrun \
  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
  --nproc_per_node=8 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./data/configs/stage2.yaml \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --model_path $MODEL_PATH \
  --num_shard 8 \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --timestep_shift 2.0 \
  --use_flex True \
  --resume-from $RESUME_FROM \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 20 \
  --save_every 16000 \
  --del_previous_state True \
  --lr 1e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-7 \
  --ce_weight 0.25 \
  --mse_weight 1 \
  --warmup_steps 500 \
  --total_steps 16000 \
  --ema 0.995 \
  --num_workers 6 \
  --expected_num_tokens 46080 \
  --max_num_tokens 51200 \
  --max_num_tokens_per_sample 25600 \
  --prefer_buffer_before 25600 \
  --text_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --debug_batches 3 \
  --enable_wandb $WANDB_ENABLE \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID \
  --wandb_resume $WANDB_RESUME \
  --wandb_offline $WANDB_OFFLINE
