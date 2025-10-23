<div align="center">

# <img src="assets/mathcanvas.png" width="64" alt="MathCanvas Logo" style="vertical-align: middle;"/> MathCanvas

## Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning

<p>
    <a href="https://mathcanvas.github.io/" target="_blank"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"></a>
    <a href="https://arxiv.org/pdf/2510.14958" target="_blank"><img src="https://img.shields.io/badge/📖-Paper-b31b1b" alt="Paper"></a>
    <a href="https://github.com/shiwk24/MathCanvas" target="_blank"><img src="https://img.shields.io/badge/💻-Code-green" alt="Code"></a>
    <a href="https://mathcanvas.github.io/#leaderboard" target="_blank"><img src="https://img.shields.io/badge/📊-Leaderboard-orange" alt="Leaderboard"></a>
    <a href="https://huggingface.co/shiwk24/BAGEL-Canvas" target="_blank"><img src="https://img.shields.io/badge/🤗-Model-yellow" alt="Model"></a>

</p>

<details>
  <summary><img src="https://img.shields.io/badge/🤗-Datasets-yellow" alt="Datasets"></summary>
  <p>
    <a href="https://huggingface.co/datasets/shiwk24/MathCanvas-Bench" target="_blank">MathCanvas-Bench</a>&ensp;·&ensp;
    <a href="https://huggingface.co/datasets/shiwk24/MathCanvas-Instruct" target="_blank">MathCanvas-Instruct</a>&ensp;·&ensp;
    <a href="https://huggingface.co/datasets/shiwk24/MathCanvas-Edit" target="_blank">MathCanvas-Edit</a>&ensp;·&ensp;
    <a href="https://huggingface.co/datasets/shiwk24/MathCanvas-Imagen" target="_blank">MathCanvas-Imagen</a>&ensp;·&ensp;
  </p>
</details>


<br>

<p>
<a href="https://scholar.google.com/citations?user=GjRC15wAAAAJ&hl=en" target="_blank">Weikang Shi</a><sup>1*</sup>,&ensp;
<a href="https://aldrichyu.github.io/" target="_blank">Aldrich Yu</a><sup>1*</sup>,&ensp;
<a href="https://rongyaofang.github.io/" target="_blank">Rongyao Fang</a><sup>1*†</sup>,&ensp;
<a href="https://scholar.google.com/citations?user=reSJxKkAAAAJ&hl=zh-CN" target="_blank">Houxing Ren</a><sup>1</sup>,&ensp;
<a href="https://wangk.org/" target="_blank">Ke Wang</a><sup>1</sup>,&ensp;
<a href="https://scholar.google.com/citations?user=cC8lXi8AAAAJ&hl=zh-CN" target="_blank">Aojun Zhou</a><sup>1</sup>,&ensp;
<a href="https://scholar.google.com/citations?user=kQ3AisQAAAAJ&hl=zh-CN" target="_blank">Changyao Tian</a><sup>1</sup>,
<br>
<a href="https://cynricfu.github.io/" target="_blank">Xinyu Fu</a><sup>2</sup>,&ensp;
<a href="https://scholar.google.com/citations?user=nk2R0cMAAAAJ&hl=en" target="_blank">Yuxuan Hu</a><sup>1</sup>,&ensp;
<a href="https://scholar.google.com/citations?user=ewuGUCwAAAAJ&hl=en" target="_blank">Zimu Lu</a><sup>1</sup>,&ensp;
<a href="https://leonhlj.github.io/" target="_blank">Linjiang Huang</a><sup>3</sup>,&ensp;
<a href="https://colalab.net/" target="_blank">Si Liu</a><sup>3</sup>,&ensp;
<a href="https://ruiliu-ai.github.io/" target="_blank">Rui Liu</a><sup>2‡</sup>,&ensp;
<a href="https://www.ee.cuhk.edu.hk/~hsli/" target="_blank">Hongsheng Li</a><sup>1‡</sup>
</p>

<p>
<sup>1</sup>MMLab, CUHK&ensp;&ensp; <sup>2</sup>Huawei Research&ensp;&ensp; <sup>3</sup>BUAA
<br>
<small><sup>*</sup>Equal Contribution&ensp;&ensp; <sup>†</sup>Project Lead&ensp;&ensp; <sup>‡</sup>Corresponding Author</small>
</p>

</div>

## 💥 News

- **[2025-10-23]** We release the [training/inference code](./BAGEL/) of BAGEL-Canvas and [evaluation scripts](./evaluation/) for MathCanvas-Bench.
- **[2025-10-18]** Our model and datasets are now accessible at [Huggingface](https://huggingface.co/collections/shiwk24/mathcanvas).
- **[2025-10-18]** Our paper is now accessible at [ArXiv Paper](https://arxiv.org/pdf/2510.14958).

## 📖 Introduction

🌟 This is the official repository for the paper **"MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning"**. This repository will host the datasets, evaluation code, and models associated with our work.

<p align="center">
  <img src="assets/teaser.jpg" alt="MathCanvas Teaser" width="100%">
</p>
<p align="center">
  <small><i>
    MathCanvas demonstrates the first successful application of intrinsic Visual Chain-of-Thought (VCoT) for complex mathematical reasoning, outperforming previous attempts.
  </i></small>
</p>

**MathCanvas** is a comprehensive framework designed to endow unified Large Multimodal Models (LMMs) with intrinsic **Visual Chain-of-Thought (VCoT)** capabilities for mathematics. Our approach enables models to strategically generate and reason with visual aids, mirroring how humans solve complex problems in domains like geometry and function analysis.

## 🚀 Model Training and Inference

For detailed instructions on setting up the environment, training the **BAGEL-Canvas** model, and running inference, please refer to our comprehensive guide:

*   **[📄 USAGE.md](./BAGEL/USAGE.md)**: The complete guide for model training and inference.

## 📊 Evaluation

This section provides instructions for evaluating model performance on our **MathCanvas-Bench** benchmark. The evaluation process relies on an LLM-based judge (GPT-4.1) to assess the correctness of the generated answers.

To evaluate the inference results on **MathCanvas-Bench**, follow the steps below:

1.  **Configure the Evaluation Script**:
    Open the `evaluation/mathcanvas_evaluate_4.1.sh` script and set the `your_api_key` and `your_base_url` variables.

2.  **Run Evaluation**:
    Execute the following command, replacing `{INFERENCE_DIR}` with the path to your inference output.

    ```bash
    cd MathCanvas/evaluation
    bash mathcanvas_evaluate_4.1.sh {INFERENCE_DIR}
    ```

3.  **View the Results**:
    After the script finishes, an evaluation summary will be generated. This summary includes detailed accuracy metrics, such as:
    *   Weighted scoring accuracy and complete accuracy.
    *   Accuracy broken down by knowledge category.
    *   Accuracy based on whether the question includes initial images.

## ✨ Highlights

### MathCanvas-Bench & MathCanvas-Instruct

To facilitate rigorous evaluation, we introduce **MathCanvas-Bench**, a challenging benchmark with 3K problems that require models to produce interleaved visual-textual solutions. The models are fine-tuned on **MathCanvas-Instruct**, a new 219K-example dataset of interleaved visual-textual reasoning paths, teaching them *when* and *how* to leverage visual aids.

<p align="center">
  <img src="assets/benchmark_statistics.jpg" alt="Benchmark Statistics" width="100%">
</p>
<p align="center">
  <small><i>
    Statistical analysis of the MathCanvas-Bench dataset.
  </i></small>
</p>

<details>
  <summary>Examples from the MathCanvas-Instruct dataset, showing interleaved visual and textual reasoning steps.</summary>
  <p align="center">
    <img src="assets/instruct_example1.jpg" alt="Instruction Example 1" width="100%" style="margin-top: 12px; margin-bottom: 12px;">
    <img src="assets/instruct_example2.jpg" alt="Instruction Example 2" width="100%" style="margin-bottom: 12px;">
    <img src="assets/instruct_example3.jpg" alt="Instruction Example 3" width="100%">
  </p>
</details>

### MathCanvas-Edit & MathCanvas-Imagen

We constructed a massive 15.2M-pair pre-training corpus to teach foundational visual manipulation skills. This includes **MathCanvas-Imagen** (10M caption-to-diagram pairs) for mastering diagram generation and **MathCanvas-Edit** (5.2M step-by-step editing trajectories) for diagram editing.

<p align="center">
  <img src="assets/pipeline.jpg" alt="Data Curation Pipeline" width="100%">
</p>
<p align="center">
  <small><i>
    The curation pipeline for the MathCanvas-Edit and MathCanvas-Imagen datasets.
  </i></small>
</p>

<details>
  <summary>Examples from the MathCanvas-Edit and MathCanvas-Imagen datasets.</summary>
  <p align="center">
    <img src="assets/edit_example1.jpg" alt="Edit Example 1" width="100%" style="margin-top: 12px; margin-bottom: 12px;">
    <img src="assets/imagen_example1.jpg" alt="Imagen Example 1" width="100%" style="margin-bottom: 12px;">
    <img src="assets/imagen_example2.jpg" alt="Imagen Example 2" width="100%">
  </p>
</details>

### Two-Stage Training Recipe

Our model, **BAGEL-Canvas**, is trained using a two-stage framework:
1.  **Stage I: Mastering Visual Manipulation:** The model learns from the 15.2M examples in MathCanvas-Imagen and MathCanvas-Edit to create and edit mathematical diagrams.
2.  **Stage II: Developing Strategic Reasoning:** The model is then trained on MathCanvas-Instruct to strategically generate visual steps as part of a solution.

<p align="center">
  <img src="assets/recipe.jpg" alt="MathCanvas Framework" width="100%">
</p>
<p align="center">
  <small><i>
    The two-stage training framework of MathCanvas.
  </i></small>
</p>


## 📝 TODO

Our code and models are currently being prepared for public release. We appreciate your patience!

- [x] Release training and inference code for **BAGEL-Canvas**.
- [x] Release evaluation scripts for the **MathCanvas-Bench**.
- [ ] Update the evaluation scripts for the **MathCanvas-Bench** to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
- [ ] Release the data generation code for **Foundational Structure Generation** in MathCanvas-Edit.

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{shi2025mathcanvasintrinsicvisualchainofthought,
      title={MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning}, 
      author={Weikang Shi and Aldrich Yu and Rongyao Fang and Houxing Ren and Ke Wang and Aojun Zhou and Changyao Tian and Xinyu Fu and Yuxuan Hu and Zimu Lu and Linjiang Huang and Si Liu and Rui Liu and Hongsheng Li},
      year={2025},
      eprint={2510.14958},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14958}, 
}
@inproceedings{
  wang2025mathcodervl,
  title={MathCoder-{VL}: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning},
  author={Ke Wang and Junting Pan and Linda Wei and Aojun Zhou and Weikang Shi and Zimu Lu and Han Xiao and Yunqiao Yang and Houxing Ren and Mingjie Zhan and Hongsheng Li},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025},
  url={https://openreview.net/forum?id=nuvtX1imAb}
}
```