# data_handlers/uni_handler.py

import os
import json
import shutil
from PIL import Image
import torch.distributed as dist
from typing import List, Dict, Any
from io import BytesIO

from .base_handler import BaseDataHandler
from padding_utils import pad_image_to_bytes_cv2

def process_mathvision(item):
    if 'options' in item and item['options']:
        question_text = item['question']
        options = item['options']

        # Build the options string
        options_str = "\nOptions:"
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            options_str += f"\n{letter}. {option}"
            
        # Append the options to the question
        item['question'] = question_text + options_str
    
    return item

class UniHandler(BaseDataHandler):
    """
    A universal data handler for a standardized dataset format.

    This handler assumes the data has been pre-processed into a unified format where:
    - 'question' is a single string containing all textual information.
    - 'image' is either a single string (relative path) for one image, or a list
      of strings for multiple images.
    
    This version includes image padding and resizing to 512x512, mirroring the
    logic from GeolauxHandler.
    """

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads data from a JSON or JSONL file. This logic is generic.
        JSONL is assumed for files not ending with .json.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                return [json.loads(line) for line in f]

    def prepare_input(self, item: Dict[str, Any], dataset_name, **kwargs) -> List:
        """
        Prepares model input by combining the question text with one or more
        padded and resized images.
        """
        image_root = kwargs.get('image_root')
        if not image_root:
            raise ValueError("`image_root` must be provided for UniHandler.")

        if dataset_name == 'mathvision':
            item = process_mathvision(item)

        inputs = [item['question']]
        image_paths = item.get('image')

        if not image_paths:
            return inputs

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for rel_path in image_paths:
            full_path = os.path.join(image_root, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        image_bytes = f.read()
                    
                    padded_image_bytes = pad_image_to_bytes_cv2(
                        image_bytes, target_size=512, padding_size=0
                    )
                    
                    padded_image = Image.open(BytesIO(padded_image_bytes)).convert('RGB')
                    inputs.append(padded_image)

                except Exception as e:
                    if dist.is_initialized() and dist.get_rank() == 0:
                        print(f"Error processing image {full_path} for item {item.get('id')}: {e}")
            else:
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Warning: Image not found at {full_path} for item {item.get('id')}")
        
        return inputs

    def save_results(self, reasoning_steps: List, item: Dict[str, Any], output_dir: str, **kwargs):
        """
        Saves all inference artifacts. This method remains unchanged as it
        operates on file paths, not image content.
        """
        image_root = kwargs.get('image_root')

        # --- 1. Save generated images and create the reasoning log ---
        generated_images_dir = os.path.join(output_dir, "images")
        os.makedirs(generated_images_dir, exist_ok=True)

        processed_steps = []
        image_counter = 0
        for step in reasoning_steps:
            processed_step = step.copy()
            if step['type'] == 'image':
                image_counter += 1
                image_filename = f"reasoning_image_{image_counter}.png"
                image_path = os.path.join(generated_images_dir, image_filename)
                processed_step['content'] = os.path.join("images", image_filename)
                step['content'].save(image_path)
            processed_steps.append(processed_step)

        # --- 2. Save the main reasoning_result.json file ---
        result_json = {
            "id": item.get('id', 'unknown_id'),
            "reasoning_steps": processed_steps,
            "summary": {
                "text_steps": len([s for s in reasoning_steps if s['type'] == 'text']),
                "image_steps": len([s for s in reasoning_steps if s['type'] == 'image']),
            }
        }
        with open(os.path.join(output_dir, "reasoning_result.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

        # --- 3. Save the original item data for reference ---
        with open(os.path.join(output_dir, "ori.json"), 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=4, ensure_ascii=False)

        # --- 4. Copy all original images used in the input ---
        ori_images_dir = os.path.join(output_dir, "ori_images")
        os.makedirs(ori_images_dir, exist_ok=True)
        
        original_image_paths = item.get('image')
        if not original_image_paths:
            return

        if isinstance(original_image_paths, str):
            original_image_paths = [original_image_paths]
        
        for img_path in original_image_paths:
            src_path = os.path.join(image_root, img_path)
            dest_path = os.path.join(ori_images_dir, os.path.basename(img_path))
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Warning: Original image not found at {src_path}")