from io import BytesIO
import random
import traceback
import PIL
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class InterleaveReasoningIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    
    def __init__(self, *args, **kwargs):
        # Call parent class initialization
        super().__init__(*args, **kwargs)
        
        # Add special tokens
        self.start_of_image = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.im_start = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.end_of_text = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')

    def _get_pil_image(self, image_data):
        """
        Elegantly handles loading an image whether it's in bytes format or already a PIL Image object.
        """
        if isinstance(image_data, bytes):
            # If the data is bytes, open it from a byte stream
            return Image.open(BytesIO(image_data))
        elif isinstance(image_data, Image.Image) or isinstance(image_data, PIL.PngImagePlugin.PngImageFile):
            # If it's already a PIL Image object, just return it
            return image_data
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            # Handle the corner case where image is a dict with a 'bytes' key
            return Image.open(BytesIO(image_data['bytes']))
        else:
            # If the type is unexpected, raise an error for easier debugging
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def parse_row(self, row):
        """Parse a single parquet row into the required format"""
        try:
            # Extract basic fields
            answer = row.get('answer', '')
            question_interleave = row.get('question_interleave', [])
            solution_interleave = row.get('solution_interleave', [])

            question_images = row.get('question_images', [])
            solution_images = row.get('solution_images', [])

            if not answer or len(question_interleave) == 0 or len(solution_interleave) == 0:
                return {}
            
            data = self._init_data()
            
            # 1. Process question part (question_interleave)
            success = self._add_interleave_sequence(
                data, 
                question_interleave,
                question_images,
                section_type="question"
            )
            if not success:  # Image loading failed
                return {}
            
            # 2. Process solution part (solution_interleave)  
            success = self._add_interleave_sequence(
                data,
                solution_interleave,
                solution_images,
                section_type="solution"
            )
            if not success:  # Image loading failed
                return {}
            
            return data
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing row: {e}")
            return {}

    def _add_interleave_sequence(self, data, interleave_list, images_list, section_type):
        """Add interleaved text and image sequence in original order"""
        
        try:
            image_count = 0
            
            for i, item in enumerate(interleave_list):
                if item['type'] == 'text':
                    text_content = item['content']
                    if text_content and text_content.strip():
                        # Determine if this text should predict special token
                        special_token_label = None
                        
                        if section_type == "question":
                            # Question texts don't need loss, no special token prediction
                            need_loss = False
                        else:  # solution
                            # Solution texts need loss and may predict special tokens
                            need_loss = True
                            
                            # Check if next item is an image
                            has_following_image = (i + 1 < len(interleave_list)) and (interleave_list[i + 1]['type'] == 'image')
                            
                            # Check if this is the last item in the sequence
                            is_last_item = (i == len(interleave_list) - 1)
                            
                            if has_following_image:
                                special_token_label = self.start_of_image
                            elif is_last_item:
                                # Last text in solution predicts end_of_text
                                special_token_label = self.end_of_text
                            else:
                                special_token_label = self.im_start
                        
                        data = self._add_text(
                            data,
                            text_content.strip(),
                            need_loss=need_loss,
                            enable_cfg=True,
                            special_token_label=special_token_label
                        )
                        
                elif item['type'] == 'image':
                    image_count += 1
                    
                    try:
                        image_index = item['index']
                        if image_index < len(images_list):
                            image_data = images_list[image_index]
                            image = self._get_pil_image(image_data)
                            image = pil_img2rgb(image)
                        else:
                            print(f"Image index {image_index} out of bounds for images_list of length {len(images_list)}")
                            return False
                            
                        if section_type == "question":
                            # Question images: no loss, but need both VAE and VIT for potential editing and understanding
                            data = self._add_image(
                                data,
                                image,
                                need_loss=False,
                                need_vae=True,
                                need_vit=True,
                                enable_cfg=True,
                            )
                        else:  # solution
                            # Solution images: always need loss, VAE, and VIT for train-inference consistency
                            data = self._add_image(
                                data,
                                image,
                                need_loss=True,
                                need_vae=True,  
                                need_vit=True,               
                                enable_cfg=True,
                            )
                                
                    except Exception as e:
                        print(f"Failed to load image with index {item.get('index', 'N/A')}: {e}")
                        traceback.print_exc()
                        return False
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing interleave sequence: {e}")
            return False