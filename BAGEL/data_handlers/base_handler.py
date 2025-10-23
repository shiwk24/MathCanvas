# data_handlers/base_handler.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDataHandler(ABC):
    """
    Abstract Base Class for data handlers. It defines the common interface
    for loading data, preparing model inputs, and saving inference results.
    
    This allows the main inference script to remain generic and delegate
    all dataset-specific operations to a concrete handler implementation.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads data from a given file path.

        Args:
            file_path (str): The path to the dataset file (e.g., a .json or .jsonl file).

        Returns:
            List[Dict[str, Any]]: A list of data items, where each item is a dictionary.
        """
        pass

    @abstractmethod
    def prepare_input(self, item: Dict[str, Any], **kwargs) -> List:
        """
        Prepares the model input list (text and PIL Images) for a single data item.

        Args:
            item (Dict[str, Any]): A single data item from the list returned by load_data.
            **kwargs: Can contain necessary paths like `image_root`.

        Returns:
            List: A list of interleaved strings and PIL.Image.Image objects.
        """
        pass

    @abstractmethod
    def save_results(self, reasoning_steps: List, item: Dict[str, Any], output_dir: str, **kwargs):
        """
        Saves all inference artifacts (generated images, JSON results, original data)
        for a single data item.

        Args:
            reasoning_steps (List): The output from the model's reasoning process.
            item (Dict[str, Any]): The original data item being processed.
            output_dir (str): The directory where results for this item should be saved.
            **kwargs: Can contain necessary paths like `image_root`
        """
        pass