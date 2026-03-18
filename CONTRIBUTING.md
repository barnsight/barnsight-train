# Contributing to BarnSight Training

First off, thank you for considering contributing to BarnSight! It's people like you that make this tool better for everyone.

## Where to Start?

The `barnsight-train` repository is focused purely on model training, hyperparameter tuning, and producing edge-ready artifacts.
Good areas for contribution include:

1. **Hyperparameter Optimization**: Finding better learning rates, augmentation strategies, or loss weights for barn environments.
2. **Model Architectures**: Testing and benchmarking different YOLO variations (e.g., YOLOv8, YOLOv10, YOLO11) or completely different architectures.
3. **Data Pipelines**: Improving how we load, filter, and preprocess farm images.
4. **Export Pipelines**: Enhancing the export scripts (e.g., converting `.pt` to `.onnx`, INT8 quantization).

## Reporting Bugs

If you find a bug in the training scripts, please open an issue and include:

* A clear and descriptive title.
* Steps to reproduce the issue.
* The exact error message or traceback.
* Your operating system, Python version, and CUDA version (if applicable).

## Suggesting Enhancements

If you have an idea for an enhancement, please open an issue describing your idea. Include:

* The motivation: Why is this enhancement useful?
* The proposed implementation: How would you achieve this?

## Pull Request Process

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies** using `uv sync`.
3.  **Make your changes**. If you are modifying training scripts, ensure they still run end-to-end.
4.  **Test your changes**. Run a short mock training session (e.g., `uv run train.py --epochs 1`) to ensure no syntax errors were introduced.
5.  **Format your code** appropriately.
6.  **Create a Pull Request**. Provide a clear description of the changes.

## Development Environment Setup

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast and reliable dependency management.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/barnsight-train.git
cd barnsight-train

# Sync the virtual environment
uv sync

# Run training
uv run train.py
```

## Guidelines

* Keep changes focused. Avoid massive, monolithic PRs.
* Document your changes. If you add a new CLI argument, add it to the `README.md` and the script's `argparse` description.
* Ensure reproducibility. Try to make sure your training experiments can be reproduced by others by fixing seeds if introducing new random behaviors.
