# Opposed Gradient Projection for Fine-Tuning LLMs

This repository contains a standalone Python script for fine-tuning Large Language Models (LLMs) like Google's Gemma using a novel technique called **Opposed Gradient Projection**. This method is designed to "unlearn" or suppress undesirable behaviors—such as refusing to answer sensitive questions—without catastrophically forgetting the model's original capabilities.

The entire implementation is contained within a single script, designed for easy execution in environments like Google Colab with GPU access (e.g., A100).

## Table of Contents
- [Core Concept](#core-concept)
- [How It Works](#how-it-works)
- [Features](#features)
- [Setup and Usage](#setup-and-usage)
- [Configuration](#configuration)
- [Monitoring with Weights & Biases](#monitoring-with-weights--biases)
- [Future Work](#future-work)

## Core Concept

Modern instruction-tuned LLMs are heavily aligned for safety, often causing them to refuse answering even harmless prompts if they seem adjacent to sensitive topics. Standard fine-tuning (SFT) on a dataset of "un-refused" answers can work, but it risks damaging the model's core alignment and helpfulness.

**Opposed Gradient Projection** offers a more precise solution. Instead of just learning what to say, it simultaneously teaches the model what *not* to say. It does this by calculating two distinct gradients during each training step and using one to steer the other. For instance in the decensoring example:

1.  **Positive Gradient (`g_sft`)**: The standard gradient that moves the model's weights to make the desired "un-refused" answer more likely.
2.  **Negative Gradient (`g_unlearn`)**: A gradient calculated by comparing the model's output to the original, "safe" but refusing answer from a frozen copy of the model. This gradient represents the direction of the undesirable behavior.

The key insight is to **project** the positive gradient onto the subspace orthogonal to the negative gradient. This effectively removes any component of the desired update that would also reinforce the undesired refusal behavior.

The final update is: `Δθ = g_sft - projection(g_sft onto g_unlearn)`.

 
*(Note: A diagram illustrating the vector projection would be a great addition here.)*

## How It Works

The script implements the following pipeline:

1.  **Model Loading**: A "student" model (the one we're training) and a "teacher" model (a frozen, unchanged copy) are loaded. The teacher acts as a stable reference for the original, safe-but-refusing behavior.
2.  **Dataset Preparation**: The script uses the `PKU-Alignment/BeaverTails` dataset, specifically filtering for examples that the base model would likely refuse (`is_safe=False`).
3.  **Hybrid Training Step**: For each batch of data:
    a. The **positive gradient (`g_sft`)** is calculated by computing the loss between the student model's logits and the desired answer's labels.
    b. The **negative gradient (`g_unlearn`)** is calculated via KL-divergence loss between the student's logits and the teacher's logits. This represents the direction towards the teacher's refusal.
    c. The projection scalar is computed: `scalar = dot(g_sft, g_unlearn) / ||g_unlearn||²`.
    d. The final gradient applied to the student model's weights is `g_final = g_sft - scalar * g_unlearn`. This update encourages the desired answer while actively suppressing the refusal behavior.
4.  **Evaluation**: At the end of each epoch, the model is evaluated on two key metrics:
    *   **Validation Loss**: To monitor for overfitting on a held-out set.
    *   **Refusal Rate**: A generative evaluation that directly measures whether the model's tendency to refuse sensitive prompts has decreased.

## Features

*   **Standalone Script**: Zero-dependency besides a `pip install` line. Perfect for Colab.
*   **Mixed-Precision Training**: Utilizes `bfloat16` for efficient training on modern GPUs (e.g., A100/H100).
*   **Layer-wise Learning Rates**: Implements a simple form of discriminative learning rates, applying smaller updates to the initial layers and larger updates to the final layers.
*   **Weights & Biases Integration**: Automatically logs training/validation loss, refusal rate, and hyperparameters for easy experiment tracking.
*   **Smoke Test**: Includes a `RUN_SMOKE_TEST` flag to quickly verify that the entire pipeline runs without errors on a tiny subset of data.

## Setup and Usage

1.  **Environment**: This script is designed for a GPU-accelerated Python environment like Google Colab.
2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/opposed-gradient-projection.git
    cd opposed-gradient-projection
    ```
3.  **Secrets/Tokens**:
    *   **Hugging Face**: You will need a Hugging Face token to download the Gemma model. Create one in your HF settings and add it as a secret named `HF_TOKEN` in your Colab notebook.
    *   **Weights & Biases (Optional)**: If you want to log experiments, you'll need a W&B API key. Add it as a secret named `WANDB_API_KEY`.
4.  **Run the Script**: Open the `.py` file in a notebook environment and run all cells. The script will automatically install dependencies, prompt for tokens if secrets are not found, and begin training.

## Configuration

All major hyperparameters are located in the "🚀 CONTROL PANEL" section at the top of the script:

*   `RUN_SMOKE_TEST`: Set to `True` to run a quick test, or `False` for the full training run.
*   `WANDB_MODE`: Set to `"online"`, `"offline"`, or `"disabled"`.
*   `USE_PROJECTION`: Must be `True` to enable the core logic of this project.
*   `UNLEARN_WEIGHT` / `POSITIVE_WEIGHT`: Control the magnitude of the opposed gradients.
*   `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, etc., in the `Config` class can be adjusted to tune the training process.

## Monitoring with Weights & Biases

This project is integrated with W&B for seamless experiment tracking. After logging in, you can monitor:

*   `avg_epoch_loss` & `avg_val_loss`: To track model learning and prevent overfitting.
*   `refusal_rate`: The key metric for this project. A downward trend indicates success.
*   `train_loss_step`: A granular, step-by-step training loss.

## Future Work

*   **Generalize to Other Models**: Adapt the script to work with other popular LLMs like Llama or Mistral.
*   **Advanced Projection Techniques**: Explore more sophisticated methods for gradient projection or surgery.
*   **Hybrid Training Datasets**: Implement a more complex data pipeline that mixes safe and unsafe prompts, applying standard SFT for safe data and opposed gradient projection for unsafe data within the same training run.

---
