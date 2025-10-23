import os
import json
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from safetensors.torch import load_file
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset

from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache


def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def set_random_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_image(prompt, gen_model, vae_model, tokenizer, new_token_ids, num_timesteps=50, cfg_scale=4.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, batch_size=4, resolution=512, device=None, enable_taylorseer=False, verbose: bool = True):
    """Generates images from a text prompt using the Bagel model."""
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * batch_size,
        new_token_ids=new_token_ids,
    )

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * batch_size
    cfg_new_rope = [0] * batch_size

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)] * batch_size,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                enable_taylorseer=enable_taylorseer,
                verbose=verbose,
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    # Model and Checkpoint Paths
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/', help='Path to the base model configuration.')
    parser.add_argument('--ckpt-path', type=str, default='hf/BAGEL-7B-MoT/', help='Path to the model checkpoint (safetensors).')
    parser.add_argument('--eval_ema', type=str, default="ema", choices=['ema', 'model'], help="Whether to evaluate the EMA model or the standard model.")

    # Data and Output
    parser.add_argument("--dataset_name", type=str, default="code_derived_captions", help="A name for this evaluation run, used in the output directory path.")
    parser.add_argument("--parquet_file", type=str, required=True, help="Parquet file containing prompts and images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--gen_num", type=int, default=50, help="Number of prompts to generate images for.")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate per prompt.")

    # Generation Parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.4, 1.0])
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--enable_taylorseer", action='store_true', help="Enable Taylorseer acceleration.")
    
    args = parser.parse_args()

    args.eval_ema = (args.eval_ema == 'ema')

    # --- Setup Distributed Environment and Seed ---
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    set_random_seed(args.seed)

    # --- Prepare Output Directory ---
    output_dir = os.path.join(
        args.output_dir, args.dataset_name,
        f"sample{args.gen_num}-size{args.resolution}-shift{args.timestep_shift}-cfg{args.cfg_scale}-steps{args.num_timesteps}-seed{args.seed}"
    )
    if args.eval_ema:
        output_dir += "--ema"
    if args.enable_taylorseer:
        output_dir += "--taylorseer"

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output images will be saved in: {output_dir}")

    # --- Load Model Components (VAE, LLM, ViT, and main Bagel model) ---
    if rank == 0: print("Loading model components...")
    
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))
    vae_model = vae_model.to(device).eval()

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
        vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    device_map = infer_auto_device_map(
        model, max_memory={rank: '80GB'}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"], dtype=torch.bfloat16
    )
    if rank == 0: print("Device map inferred:", device_map)

    model_state_dict_path = os.path.join(args.ckpt_path, "ema.safetensors") if args.eval_ema else os.path.join(args.ckpt_path, "model.safetensors")
    load_checkpoint_and_dispatch(
        model, checkpoint=model_state_dict_path, device_map=device_map,
        offload_buffers=True, dtype=torch.bfloat16, force_hooks=True,
    )
    if rank == 0: print(f"Successfully loaded weights from {model_state_dict_path}")

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    gen_model = model.eval()

    # --- Load and Prepare Dataset ---
    if rank == 0: print(f"Loading dataset from {args.parquet_file}...")
    
    dataset = load_dataset("parquet", data_files=args.parquet_file, split="train")
    
    # Ensure 'id' column exists. If not, create it.
    if 'id' not in dataset.column_names:
        dataset = dataset.map(lambda example, idx: {'id': f"idx_{idx:05d}"}, with_indices=True)

    # Sample a subset of the data if requested
    if args.gen_num < len(dataset):
        dataset = dataset.shuffle(seed=0).select(range(args.gen_num))

    total_prompts = len(dataset)
    if rank == 0: print(f"Dataset loaded. Total prompts to process: {total_prompts}")

    # --- Distribute Data Across GPUs and Run Inference ---
    prompts_per_gpu = (total_prompts + world_size - 1) // world_size
    start_idx = rank * prompts_per_gpu
    end_idx = min(start_idx + prompts_per_gpu, total_prompts)
    
    if rank == 0:
        print(f"World size: {world_size}. Each GPU will process approximately {prompts_per_gpu} prompts.")
    print(f"GPU {rank}: Processing {end_idx - start_idx} prompts (indices {start_idx} to {end_idx - 1})")

    # The main generation loop
    for i in tqdm(range(start_idx, end_idx), desc=f"GPU {rank} Inference"):
        set_random_seed(args.seed + i)
        
        metadata = dataset[i]
        prompt = metadata['caption']
        
        outpath = os.path.join(output_dir, metadata['id'])
        os.makedirs(outpath, exist_ok=True)
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # Check if images already exist to allow resuming
        if os.path.exists(os.path.join(sample_path, f"{args.num_images - 1:05}.png")):
            print(f"GPU {rank} skipping generation for {metadata['id']} as files already exist.")
            continue

        # Save metadata and original image
        with open(os.path.join(outpath, "metadata.json"), "w", encoding="utf-8") as f:
            # Convert PIL Image to a placeholder string for JSON serialization
            json_metadata = {k: v if k != 'image' else "[Image Bytes]" for k, v in metadata.items()}
            json.dump(json_metadata, f, indent=4)
        
        if 'image' in metadata and metadata['image'] is not None:
            try:
                original_image = metadata['image'] # This is a PIL.Image object
                original_image.save(os.path.join(sample_path, "ori.png"))
            except Exception as e:
                print(f"Warning: Could not save original image for {metadata['id']}. Error: {e}")

        # Generate images in batches
        generated_images = []
        num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
        for _ in range(num_batches):
            batch_images = generate_image(
                prompt=prompt, gen_model=gen_model, vae_model=vae_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, num_timesteps=args.num_timesteps, cfg_scale=args.cfg_scale,
                cfg_interval=args.cfg_interval, cfg_renorm_min=args.cfg_renorm_min,
                timestep_shift=args.timestep_shift, batch_size=args.batch_size,
                resolution=args.resolution, device=device, enable_taylorseer=args.enable_taylorseer,
                verbose=(rank == 0) # Only print progress from rank 0
            )
            generated_images.extend(batch_images)

        # Save generated images
        for j, sample_img in enumerate(generated_images[:args.num_images]):
            # Optional: crop the bounding box to remove black borders
            try:
                sample_img = sample_img.crop(sample_img.getbbox())
            except TypeError: # Happens if the image is solid black
                pass
            sample_img.save(os.path.join(sample_path, f"{j:05}.png"))

    print(f"GPU {rank} has completed all its tasks.")
    dist.barrier()
    if rank == 0:
        print("All GPUs finished. Inference complete.")

if __name__ == "__main__":
    main()