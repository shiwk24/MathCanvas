import sys
import os
import argparse
import cv2
import numpy as np
from tqdm.auto import tqdm
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from multiprocessing import Pool, cpu_count
from functools import partial


CV2_SAMPLING_METHODS = {
    'lanczos': cv2.INTER_LANCZOS4,
    'bicubic': cv2.INTER_CUBIC,
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'area': cv2.INTER_AREA
}

def pil_img2rgb(image: Image.Image) -> Image.Image:
    """
    Converts an image that may have transparency to an RGB image with a white background using PIL.
    """
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Create a white background
        background = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        # To use the alpha channel as a mask, ensure the image is in RGBA mode
        rgba_image = image.convert("RGBA")
        # Paste the original image content onto the white background, using the alpha channel as a mask
        background.paste(rgba_image, mask=rgba_image.split()[3])
        return background
    else:
        return image.convert("RGB")

def pad_image_to_bytes_cv2(
    image_bytes: bytes,
    preferred_upscale_method: str = 'bicubic',
    preferred_downscale_method: str = 'area',
    target_size: int = 512,
    padding_size: int = 0,
    background_color_rgb: tuple = (255, 255, 255)
) -> bytes:
    """
    Processes an image using OpenCV and returns it as PNG-formatted bytes.
    
    Args:
        padding_size: The size of the padding to add to each side of the final image.
    
    Returns:
        bytes: The processed image data in PNG format, or None on failure.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img_pil:
            img_rgb_pil = pil_img2rgb(img_pil)
    except Exception as e:
        print(f"Error: PIL could not open or process the image bytes: {e}")
        return None

    img_rgb_np = np.array(img_rgb_pil)
    img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    original_height, original_width = img_bgr.shape[:2]

    # Calculate the effective target size (subtracting padding)
    effective_target_size = target_size - 2 * padding_size
    
    # Prevent effective_target_size from being too small
    if effective_target_size <= 0:
        print(f"Error: padding_size ({padding_size}) is too large for the target_size ({target_size}).")
        return None
    
    max_dimension = max(original_width, original_height)
    
    # If the original image is already a square matching the effective size, process it directly
    if max_dimension == effective_target_size and original_height == original_width:
        # Add padding directly around the original image
        background_color_bgr = background_color_rgb[::-1]
        canvas_bgr = np.full((target_size, target_size, 3), background_color_bgr, dtype=np.uint8)
        canvas_bgr[padding_size:padding_size+original_height, padding_size:padding_size+original_width] = img_bgr
        
        success, encoded_image = cv2.imencode('.png', canvas_bgr)
        if not success:
            print("Error: Failed to encode the image to PNG bytes.")
            return None
        return encoded_image.tobytes()

    # Calculate the scaling ratio based on the effective target size
    ratio = max_dimension / float(effective_target_size)
    
    # Prevent division by zero if ratio is 0
    if ratio == 0:
        return None
        
    new_width = int(original_width / ratio)
    new_height = int(original_height / ratio)

    # Choose the optimal interpolation method based on whether we are upscaling or downscaling
    if new_width > original_width or new_height > original_height:
        interpolation = CV2_SAMPLING_METHODS.get(preferred_upscale_method, cv2.INTER_CUBIC)
    else:
        interpolation = CV2_SAMPLING_METHODS.get(preferred_downscale_method, cv2.INTER_AREA)
        
    resized_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=interpolation)

    # Create a canvas and fill it with the background color (OpenCV colors are BGR)
    background_color_bgr = background_color_rgb[::-1]
    canvas_bgr = np.full((target_size, target_size, 3), background_color_bgr, dtype=np.uint8)

    # Center the resized image on the canvas
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    canvas_bgr[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_bgr

    # Encode the final BGR canvas into PNG-formatted bytes
    success, encoded_image = cv2.imencode('.png', canvas_bgr)
    
    if not success:
        print("Error: Failed to encode the processed image to PNG bytes.")
        return None

    return encoded_image.tobytes()