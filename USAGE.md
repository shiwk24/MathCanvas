# 🚀 Model Usage: Training & Inference

This document provides detailed instructions on how to set up the environment, train the **BAGEL-Canvas** model, and run inference. All commands should be run from the `BAGEL/` subdirectory.

## ⚙️ Environment Setup

First, clone the repository and set up the Conda environment.

```bash
git clone https://github.com/shiwk24/MathCanvas.git
cd MathCanvas/BAGEL

conda create -n bagel-canvas python=3.10 -y
conda activate bagel-canvas

pip install -r requirements.txt

# Note: Refer to the official FlashAttention repository for suitable CUDA version.
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## 🧠 Training

The training process is divided into two stages. The workflow for both is very similar and is outlined below.

1.  **Download Datasets**:
    Download the required datasets from our [Hugging Face Collection](https://huggingface.co/collections/shiwk24/mathcanvas).
    *   For **Stage I**, you will need `MathCanvas-Imagen` and `MathCanvas-Edit`.
    *   For **Stage II**, you will need `MathCanvas-Instruct`.

2.  **Configure Dataset Paths**:
    Modify the file `data/dataset_info.py`. In this file, replace the placeholder paths with the actual location of your downloaded datasets (referred to as `your_data_path`).

3.  **Configure and Run Training Script**:
    Modify the corresponding training script for the stage you are running. In the script, you must set the following variables:
    *   `your_model_path`: Path to the base model (for Stage I) or the Stage I checkpoint (for Stage II).
    *   `your_save_path`: Directory where the new model checkpoints will be saved.
    *   `your_wandb_key`: Your Weights & Biases API key for experiment logging.

    Once configured, launch the training using the appropriate command:

    *   **For Stage I (Visual Manipulation):**
        ```bash
        bash scripts/train/stage1.sh
        ```

    *   **For Stage II (Strategic Visual-Aided Reasoning):**
        ```bash
        bash scripts/train/stage2.sh
        ```

## 💡 Inference

This section covers how to run inference for both interleaved reasoning and text-to-image generation.

### 1. Interleaved Reasoning Inference

This inference mode, primarily driven by `mathcanvas_interleave_reasoner.py`, generates step-by-step visual and textual solutions.

**On MathCanvas-Bench:**

1.  **Get Data**: Download `MathCanvas-Bench` from [Hugging Face](https://huggingface.co/datasets/shiwk24/MathCanvas-Bench) or use the local copy provided in `MathCanvas/benchmark/` (`data.jsonl` and the `images/` directory).
2.  **Run Inference**: Edit `scripts/inference/infer_mathcanvas.sh` to set `your_model_path` and `your_save_path`, then execute:

    ```bash
    # Run inference on MathCanvas-Bench (using 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/inference/infer_mathcanvas.sh
    ```

**On Other Benchmarks (e.g., [MathVision](https://github.com/mathllm/MATH-V)):**

1.  **Run Inference**: We provide scripts for other standard benchmarks. Similarly, edit `scripts/inference/infer_mathvision.sh` with your model and save paths, then run:

    ```bash
    # Run inference on MathVision (using 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/inference/infer_mathvision.sh
    ```
    *Note: For benchmarks like MathVision, the script automatically preprocesses images by padding and resizing them to 512x512 using `padding_utils.py`.*

### 2. Imagen (Text-to-Image) Inference

This mode, based on `mathcanvas_imgen_inferencer.py`, evaluates text-to-image generation on the validation sets of the `MathCanvas-Imagen` datasets.

1.  **Configuration**: Modify the configuration section in `scripts/inference/infer_imagen.sh`:
    *   `your_data_path`: Path to the dataset.
    *   `your_model_path`: Path to your trained model checkpoint.
    *   `your_save_path`: Directory to save results.
    *   `dataset_name`: The specific dataset to evaluate. Options include `code_derived_captions`, `repurposing_competition`, `repurposing_foundational`, `mavis`, `tr_cot`.

2.  **Run Inference**:

    ```bash
    # Run text-to-image inference (using 1 GPU)
    CUDA_VISIBLE_DEVICES=0 bash scripts/inference/infer_imagen.sh
    ```

### 3. Editing Inference

For examples of diagram editing inference, please refer to the Jupyter Notebook: `inference.ipynb`.