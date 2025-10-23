import os
import json
import shutil
from datetime import datetime
from copy import deepcopy
from typing import List, Union
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
import traceback

import torch
import torch.distributed as dist
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

from data_handlers import MathCanvasHandler, UniHandler

def timestamp() -> str:
    dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    return dt_string  

def setup_distributed():
    """Initializes the distributed environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank

def get_synchronized_timestamp():
    if dist.get_rank() == 0:
        timestamp_str = timestamp()
        print('Current Timestamp :', timestamp_str)
    else:
        timestamp_str = None
    
    timestamp_list = [timestamp_str]
    dist.broadcast_object_list(timestamp_list, src=0)
    
    return timestamp_list[0]

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

    
def setup_model(checkpoint_dir: str, rank: int, checkpoint_file: str, model_path: str):
    """Loads and dispatches the model across devices."""
    device = f"cuda:{rank}"
    if rank == 0:
        print(f"Loading model from {checkpoint_dir}")
        print(f"Available GPUs for parallel execution: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    # --- Load Model Configurations ---
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # Load VAE model directly onto the target device for this process.
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    vae_model = vae_model.to(device).eval()

    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config,
        vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )
    
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 512, 14)

    max_mem_per_gpu = "80GiB"
    if rank == 0:
        print("Setting up device mapping...")

    device_map = infer_auto_device_map(
        model, max_memory={rank: max_mem_per_gpu},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"], dtype=torch.bfloat16,
    )
    if rank == 0:
        print(f"Device map for rank 0 (others will be similar): {device_map}")

    same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']

    first_device = device_map.get(same_device_modules[0], device)
    for k in same_device_modules:
        device_map[k] = first_device if k in device_map else device

    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=f"/tmp/offload_rank_{rank}" 
    )
    model = model.eval()

    dist.barrier()
    if rank == 0: 
        print('Model loaded successfully across all processes!')
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class InterleaveReasoner:
    """A class to handle the multi-step reasoning process."""
    def __init__(self, inferencer: InterleaveInferencer):
        self.inferencer = inferencer
        self.tokenizer = inferencer.tokenizer
        self.new_token_ids = inferencer.new_token_ids

        self.action_token_map = {
            self.new_token_ids['start_of_image']: 'image',
            self.new_token_ids['bos_token_id']: 'text',
            self.new_token_ids['end_of_text']: 'end',
        }
        if dist.get_rank() == 0:
            print("InterleaveReasoner initialized on each process.")

    def generate_text_with_next_token_check(self, gen_context, max_length=2048, do_sample=True, temperature=0.3):
        """Generates text and determines the next action based on the last token."""
        generated_ids = self.inferencer.gen_text(
            gen_context, max_length=max_length, do_sample=do_sample,
            temperature=temperature, return_ids=True
        )

        next_action_token = generated_ids[-1]
        text_token_ids = generated_ids[1:-2] # Exclude BOS, EOS, and NEXT_ACTION
        next_action = self.action_token_map.get(next_action_token, 'undefined')

        decoded_text = self.tokenizer.decode(text_token_ids).strip()

        return decoded_text, next_action

    @torch.no_grad()
    def reasoning_inference(self, inputs: List[Union[str, Image.Image]], system_prompt: str = None, max_iterations: int = 10, verbose_iter: bool = False, verbose_image: bool = False, **inference_kwargs):
        reasoning_steps = []
        
        gen_context = self.inferencer.init_gen_context()
        cfg_text_context = self.inferencer.init_gen_context()
        cfg_img_context = self.inferencer.init_gen_context()
        
        image_shapes = (512, 512)
        
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if system_prompt:
                gen_context = self.inferencer.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.inferencer.update_context_text(system_prompt, cfg_img_context)
            
            for input_item in inputs:
                if isinstance(input_item, str):
                    gen_context = self.inferencer.update_context_text(input_item, gen_context)
                    cfg_img_context = self.inferencer.update_context_text(input_item, cfg_img_context)
                elif isinstance(input_item, Image.Image):
                    processed_image = self.inferencer.vae_transform.resize_transform(pil_img2rgb(input_item))
                    image_shapes = processed_image.size[::-1]
                    
                    gen_context = self.inferencer.update_context_image(processed_image, gen_context, vae=True, vit=True)
                    cfg_text_context = self.inferencer.update_context_image(processed_image, cfg_text_context, vae=True, vit=True)
            
            current_mode = 'text'
            with tqdm(total=max_iterations, desc="Reasoning Steps", leave=False, disable=(dist.get_rank() != 0 or not verbose_iter)) as pbar:
                for iteration in range(max_iterations):
                    if current_mode == 'text':
                        pbar.set_description(f"Step {iteration + 1}/{max_iterations}: Generating text")

                        generated_text, next_action = self.generate_text_with_next_token_check(gen_context, do_sample=inference_kwargs.get('do_sample', True), temperature=inference_kwargs.get('text_temperature', 0.3))

                        gen_context = self.inferencer.update_context_text(generated_text, gen_context)
                        cfg_img_context = self.inferencer.update_context_text(generated_text, cfg_img_context)
                        
                        if next_action == 'end':
                            reasoning_steps.append({'type': 'text', 'content': generated_text, 'iteration': iteration + 1})
                            pbar.update(1)
                            break
                        else:
                            reasoning_steps.append({'type': 'text', 'content': generated_text, 'iteration': iteration + 1})
                            current_mode = 'image' if next_action == 'image' else 'text'

                    
                    elif current_mode == 'image':
                        pbar.set_description(f"Step {iteration + 1}/{max_iterations}: Generating image")
                        try:
                            gen_image_kwargs = {k: v for k, v in inference_kwargs.items() if k.startswith('cfg_') or k in ['timestep_shift', 'num_timesteps', 'enable_taylorseer']}
                            
                            generated_image = self.inferencer.gen_image(
                                image_shapes, gen_context,
                                cfg_text_precontext=cfg_text_context, 
                                cfg_img_precontext=cfg_img_context,
                                verbose=(verbose_image and dist.get_rank() == 0),
                                **gen_image_kwargs
                            )
                            reasoning_steps.append({'type': 'image', 'content': generated_image, 'iteration': iteration + 1})
                            
                            processed_image = self.inferencer.vae_transform.resize_transform(pil_img2rgb(generated_image))
                            gen_context = self.inferencer.update_context_image(processed_image, gen_context, vae=True, vit=True)
                            cfg_text_context = self.inferencer.update_context_image(processed_image, cfg_text_context, vae=True, vit=True)
                            
                            current_mode = 'text'
                        except Exception as e:
                            traceback.print_exc()
                            print(f"Rank {dist.get_rank()} error during image generation: {e}")
                            current_mode = 'text' 
                    
                    pbar.update(1)
        
        return reasoning_steps


if __name__ == "__main__":
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run batch inference for the BAGEL model.", epilog="If edited images appear blurry, try `global` CFG-Renorm, decrease `cfg_renorm_min` or decrease `cfg_scale`.")
    parser.add_argument(
        "--dataset_type", type=str, required=True, choices=['mathcanvas', 'uni'], 
        help="Type of the dataset to process. This determines which data handler is used."
    )
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset being processed. E.g., 'mathcanvas', 'mathvision'.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing the model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results.")
    parser.add_argument("--checkpoint_file", type=str, default="model.safetensors", help="Name of the checkpoint file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model files (configs, tokenizer).")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for the primary images.")
    
    parser.add_argument("--add_timestamp", type=str, choices=['true', 'false'], default='true', help="Whether to append a timestamp to the output directory. If set, cannot continue from previous runs.")
    parser.add_argument("--run_description", type=str, default="", help="Optional description for this inference run.")
    parser.add_argument("--verbose_iter", action='store_true', help="Enable verbose progress bars for reasoning steps.")
    parser.add_argument("--verbose_image", action='store_true', help="Enable verbose progress bars for image generation.")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum reasoning iterations per input item.")
    parser.add_argument("--skip_completed", action='store_true', help="Skip items that have already been processed and have a result file.")

    # Inference hyperparameters
    parser.add_argument("--do_sample", type=str, choices=['true', 'false'], default='true', help="Whether to use sampling for text generation. If False, greedy decoding is used.")
    parser.add_argument("--text_temperature", type=float, default=0.3, help="Sampling temperature for text generation. Higher values yield more diverse outputs.")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="Controls how strongly the model follows the text prompt. `1.0` disables text guidance. Typical range: `4.0--8.0`.")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0, help="Controls how much the model preserves input image details. `1.0` disables image guidance. Typical range: `1.0--2.0`.")
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0], help="Fraction of denoising steps where CFG is applied. Later steps can skip CFG to reduce computation. Typical: `[0.4, 1.0]`.")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Shifts the distribution of denoising steps. Higher values allocate more steps at the start (affects layout); lower values allocate more at the end (improves details).")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Total denoising steps. Typical: `50`.")

    parser.add_argument("--cfg_renorm_min", type=float, default=0.0, help="Minimum value for CFG-Renorm. `1.0` disables renorm. Typical: `0`.")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel", choices=["global", "channel", "text_channel"], help="CFG-Renorm method. `global`: Normalize over all tokens and channels (default for T2I). `channel`: Normalize across channels for each token. `text_channel`: Like `channel`, but only applies to text condition (good for editing, may cause blur).")
    parser.add_argument("--enable_taylorseer", type=str, choices=['true', 'false'], default='false', help="Whether to enable TaylorSeer for image generation.")
    

    args = parser.parse_args()

    args.add_timestamp = args.add_timestamp.lower() == 'true'
    args.do_sample = args.do_sample.lower() == 'true'
    args.enable_taylorseer = args.enable_taylorseer.lower() == 'true'
    
    # --- 2. Environment and Handler Setup ---
    set_random_seed(42)
    rank = setup_distributed()
    world_size = dist.get_world_size()

    if args.dataset_type == 'mathcanvas':
        args.dataset_name = 'mathcanvas'
        handler = MathCanvasHandler()
    elif args.dataset_type == 'uni':
        handler = UniHandler()
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    # --- 3. Configuration ---
    INFERENCE_HYPERPARAMS = dict(
        do_sample=args.do_sample, text_temperature=args.text_temperature, cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale, cfg_interval=args.cfg_interval, timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps, cfg_renorm_min=args.cfg_renorm_min, cfg_renorm_type=args.cfg_renorm_type,
        enable_taylorseer=args.enable_taylorseer,
    )
    if rank == 0:
        print("Inference hyperparameters:")
        for k, v in INFERENCE_HYPERPARAMS.items():
            print(f"  {k}: {v}")

    # --- 4. Output Path and Model Setup ---
    output_path = os.path.join(args.output_dir, args.dataset_name)
    
    if args.run_description:
        output_path = f"{output_path}_{args.run_description}"

    if args.add_timestamp:
        sync_timestamp = get_synchronized_timestamp()
        output_path = f"{output_path}_{sync_timestamp}"
    
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        print(f"All inference results will be saved under: {output_path}")

        args_file_path = os.path.join(output_path, "inference_args.json")
        current_args_dict = vars(args)

        if os.path.exists(args_file_path):
            print(f"Found existing arguments file: {args_file_path}. Verifying consistency...")
            try:
                with open(args_file_path, 'r', encoding='utf-8') as f:
                    previous_args_dict = json.load(f)

                comp_prev = previous_args_dict.copy()
                comp_curr = current_args_dict.copy()
                
                for key in ['verbose_iter', 'verbose_image']:
                    comp_prev.pop(key, None)
                    comp_curr.pop(key, None)

                if comp_prev != comp_curr:
                    print("\nERROR: Argument mismatch! Current arguments do not match the saved ones for this run.", flush=True)

                    print("Previous arguments:")
                    for k, v in previous_args_dict.items():
                        print(f"  {k}: {v}")
                    print("\nCurrent arguments:")
                    for k, v in current_args_dict.items():
                        print(f"  {k}: {v}")

                    print("Please use the original arguments or specify a new output directory.", flush=True)
                    dist.barrier()
                    exit(1)
                else:
                    print("Arguments are consistent. Resuming inference run.")

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse existing arguments file: {e}. Overwriting.")
                with open(args_file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_args_dict, f, indent=4)
        else:
            print("No existing arguments file found. Saving current arguments.")
            with open(args_file_path, 'w', encoding='utf-8') as f:
                json.dump(current_args_dict, f, indent=4)

    dist.barrier() 

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = setup_model(
        args.checkpoint_dir, rank, args.checkpoint_file, args.model_path
    )
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    reasoner = InterleaveReasoner(inferencer)

    # --- 5. Data Loading and Distribution ---
    all_eval_data = []
    if rank == 0:
        print(f"Using '{args.dataset_type}' handler to load data from {args.input_path}...")
        all_eval_data = handler.load_data(args.input_path)
        print(f"Loaded {len(all_eval_data)} total items.")

        if args.skip_completed:
            print("Filtering out already completed items...")
            initial_count = len(all_eval_data)
            
            # This list comprehension filters the data based on the existence of the result file.
            all_eval_data = [
                item for item in all_eval_data
                if not os.path.exists(
                    os.path.join(output_path, str(item.get('id')), "reasoning_result.json")
                )
            ]
            
            final_count = len(all_eval_data)
            print(f"Skipped {initial_count - final_count} completed items. {final_count} items remaining.")

    # Broadcast the filtered list of data to all processes.
    data_to_distribute = [all_eval_data] if rank == 0 else [None]
    dist.broadcast_object_list(data_to_distribute, src=0)
    eval_data_to_process = data_to_distribute[0]

    # Each process determines its own slice of the *remaining* data.
    total_items = len(eval_data_to_process)
    items_per_gpu = (total_items + world_size - 1) // world_size
    start_index = rank * items_per_gpu
    end_index = min(start_index + items_per_gpu, total_items)
    local_eval_data = eval_data_to_process[start_index:end_index]

    print(f"Rank {rank}/{world_size}: Processing {len(local_eval_data)} items (from index {start_index} to {end_index-1} of remaining tasks)")
    dist.barrier()

    inference_bar = tqdm(local_eval_data, desc=f"Rank {rank} Inference", position=rank) if not args.verbose_iter else tqdm(local_eval_data, desc=f"Rank {rank} Inference", disable=(rank != 0))

    for item in inference_bar:
        item_id = item.get('id', f"item_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
        item_output_dir = os.path.join(output_path, str(item_id))
        
        if rank == 0:
            os.makedirs(item_output_dir, exist_ok=True)
        dist.barrier()

        try:
            inputs = handler.prepare_input(item, dataset_name=args.dataset_name, image_root=args.image_root)

            reasoning_steps = reasoner.reasoning_inference(
                inputs, system_prompt=None,
                verbose_iter=args.verbose_iter, verbose_image=args.verbose_image, 
                max_iterations=args.max_iterations,
                **INFERENCE_HYPERPARAMS
            )

            handler.save_results(
                reasoning_steps, item, item_output_dir, 
                image_root=args.image_root,
            )

        except Exception as e:
            print(f"Rank {rank} failed to process item {item_id}. Error: {e}")
            traceback.print_exc()
            continue

    print(f"Rank {rank} has completed all its assigned tasks.")
    
    dist.barrier()
    
    if rank == 0:
        print("\nBatch inference completed across all GPUs!")