# data_handlers/mathcanvas_handler.py

import os
import json
import shutil
from PIL import Image
import torch.distributed as dist

from .base_handler import BaseDataHandler

class MathCanvasHandler(BaseDataHandler):
    """Data handler for the MathCanvas dataset format."""

    def load_data(self, file_path: str) -> list:
        """
        Loads data from a JSON or JSONL file.
        JSONL is assumed for files not ending with .json.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:  # Assume JSONL format for other extensions
                return [json.loads(line) for line in f]

    def prepare_input(self, item: dict, **kwargs) -> list:
        """
        Prepares model input for a single mathcanvas item by parsing the
        'question_interleave' field.
        """
        image_root = kwargs.get('image_root')
        if not image_root:
            raise ValueError("`image_root` must be provided for MathCanvasHandler.")

        inputs = []
        # The input is defined by the 'question_interleave' list
        for step in item['question_interleave']:
            if step['type'] == 'text':
                inputs.append(step['content'])
            elif step['type'] == 'image':
                img_path = os.path.join(image_root, step['content'])
                inputs.append(Image.open(img_path).convert("RGB"))
        return inputs

    def save_results(self, reasoning_steps: list, item: dict, output_dir: str, **kwargs):
        """Saves inference results for a mathcanvas item."""
        image_root = kwargs.get('image_root')

        # --- Save generated images and create reasoning log ---
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
                # Save the relative path in the JSON for portability
                processed_step['content'] = os.path.join("images", image_filename)
                step['content'].save(image_path)
            processed_steps.append(processed_step)

        # --- Save JSON results ---
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

        # Save the original item data for reference
        with open(os.path.join(output_dir, "ori.json"), 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=4, ensure_ascii=False)

        # --- Copy all original images mentioned in the item ---
        ori_images_dir = os.path.join(output_dir, "ori_images")
        os.makedirs(ori_images_dir, exist_ok=True)
        
        original_image_paths = set()
        # Collect image paths from both question and solution parts for completeness
        for part in ['question_interleave', 'solution_interleave']:
            for step in item.get(part, []):
                if step['type'] == 'image':
                    path = step['content']
                    if path:
                        original_image_paths.add(path)
        
        for img_path in original_image_paths:
            src_path = os.path.join(image_root, img_path)
            dest_path = os.path.join(ori_images_dir, os.path.basename(img_path))
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                # Only print warning from the main process
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Warning: Original image not found at {src_path}")