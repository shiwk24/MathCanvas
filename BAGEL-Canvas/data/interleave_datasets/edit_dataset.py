# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import random
import PIL
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

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
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()

        start_image_data = row["image_list"][start_idx]
        start_image = self._get_pil_image(start_image_data)
        data = self._add_image(
            data, 
            pil_img2rgb(start_image),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.0: # concat multiple instructions
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction_source = row["instruction_list"][idx-1]
                if isinstance(instruction_source, list):
                    instruction += random.choice(instruction_source) + ". "
                elif isinstance(instruction_source, str):
                    instruction += instruction_source + ". "
            
            data = self._add_text(data, instruction.rstrip(), need_loss=False)

            end_image_data = row["image_list"][end_idx]
            end_image = self._get_pil_image(end_image_data)
            data = self._add_image(
                data, 
                pil_img2rgb(end_image),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction_source = row["instruction_list"][idx-1]
                if isinstance(instruction_source, list):
                    instruction = random.choice(instruction_source)
                elif isinstance(instruction_source, str):
                    instruction = instruction_source
                
                data = self._add_text(data, instruction, need_loss=False)

                image_data = row["image_list"][idx]
                image = self._get_pil_image(image_data)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(image),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(image),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data