# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import json
import traceback
import pyarrow.parquet as pq
import random
import PIL
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

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
    
    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_data = row['image']
                                raw_image = self._get_pil_image(image_data)
                                image = pil_img2rgb(raw_image)
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            if 'captions' in row and 'caption' not in row:
                                caption_list = row['captions']

                                caps_token = [self.tokenizer.encode(v) for v in caption_list]
                                if len(caps_token) == 0:
                                    print(f'no captions in rg#{row_group_id}, {parquet_file_path}')
                                    continue
                                else:
                                    caption_token = random.choice(caps_token)
                            elif "caption" not in row or not row['caption']:
                                    print(f'Error caption in rg#{row_group_id}, {parquet_file_path}')
                                    continue
                            else:
                                caption_token = self.tokenizer.encode(row['caption']) 

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
