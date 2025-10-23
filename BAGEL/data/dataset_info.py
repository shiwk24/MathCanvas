# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .interleave_datasets import InterleaveReasoningIterableDataset

from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'competition_level_mining': UnifiedEditIterableDataset,
    'foundational_structure_generation': UnifiedEditIterableDataset,
    
    'code_derived_captions': T2IIterableDataset,
    'repurposing_competition': T2IIterableDataset,
    'others': T2IIterableDataset,
    
    "interleave_reasoning": InterleaveReasoningIterableDataset,


    't2i_pretrain': T2IIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
}


DATASET_INFO = {
    'competition_level_mining':{
        'competition_level_mining': {
            'data_dir': 'your_data_path/MathCanvas-Edit/data/competition_level_mining/train',
            'num_files': 911,
            'parquet_info_path': 'your_data_path/MathCanvas-Edit/data/competition_level_mining/train_parquet_info.json',
            "num_total_samples": 4249400,
		},
    },
    'foundational_structure_generation':{
        'foundational_structure_generation': {
            'data_dir': 'your_data_path/MathCanvas-Edit/data/foundational_structure_generation/train',
            'num_files': 187,
            'parquet_info_path': 'your_data_path/MathCanvas-Edit/data/foundational_structure_generation/train_parquet_info.json',
            "num_total_samples": 995000,
		},
    },

    'code_derived_captions': {
        'code_derived_captions': {
            'data_dir': "your_data_path/MathCanvas-Imagen/data/code_derived_captions/train",
            'num_files': 254, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 4058009,
        },
    },
    'repurposing_competition': {
        'repurposing_competition': {
            'data_dir': "your_data_path/MathCanvas-Imagen/data/repurposing_competition/train",
            'num_files': 337, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 4272979,
        },
    },
    'others': {
        'repurposing_foundational': {
            'data_dir': "your_data_path/MathCanvas-Imagen/data/repurposing_foundational/train",
            'num_files': 72, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1207065,
        },
        'mavis': {
            'data_dir': "your_data_path/MathCanvas-Imagen/data/mavis/train",
            'num_files': 78, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 513199,
        },
        'tr_cot': {
            'data_dir': "your_data_path/MathCanvas-Imagen/data/tr_cot/train",
            'num_files': 9, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 99132,
        },
    },
    
    'interleave_reasoning':{
        'mathcanvas_instruct': {
            'data_dir': "your_data_path/MathCanvas-Instruct/data/train",
            'num_files': 49,
            'parquet_info_path': "your_data_path/MathCanvas-Instruct/data/train_parquet_info.json",
            "num_total_samples": 218604,
        },
    },


    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}