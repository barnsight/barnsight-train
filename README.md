# BarnSight Training Pipeline

A professional, reproducible environment for training and optimizing computer vision models for on-device animal excrement detection.

This repository encompasses the training and experimentation architecture for the BarnSight project. Its primary responsibility is the orchestration of data, hyperparameter tuning, and model compilation to yield robust artifacts suitable for edge deployment.

---

## Overview

Agricultural environments present complex visual challenges:
- Variable and inconsistent lighting
- Unpredictable floor conditions and materials
- Multi-species subject tracking
- High incidence of visual occlusion

The BarnSight training pipeline is designed to address these variables by providing a structured, configuration-driven approach. This repository does not host the runtime service; rather, it produces the optimized model artifacts (e.g., `.pt`, `.onnx`) required by edge inference devices.

## Getting Started

### Prerequisites

Dependency management and environment isolation are handled via [`uv`](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/barnsight/barnsight-train.git barnsight-train
cd barnsight-train

# Synchronize the environment
uv sync
```

### Dataset Specification

The pipeline expects data to be formatted according to standard YOLO specifications. Ensure your dataset is structured as follows, with the configuration mapped to `datasets/training/data.yaml`:

```text
datasets/training/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## Training Lifecycle

The model development process follows a standardized lifecycle:

1. **Dataset Preparation:** Curate and label imagery representing diverse environmental conditions and contamination levels.
2. **Configuration:** Define architecture and hyperparameters within `settings.ini`.
3. **Execution:** Initiate training runs locally or via cloud instances (e.g., Google Colab).
4. **Artifact Export:** Harvest optimized model weights from `datasets/result/train/weights/` for subsequent edge deployment.

### Execution

To initiate a training run utilizing the default configuration specified in `settings.ini`:

```bash
uv run train.py
```

For rapid experimentation, parameters may be overridden directly via the command line interface:

```bash
uv run train.py --epochs 50 --batch 16 --model yolo11s.pt --imgsz 640
```

## Configuration Architecture

System behavior is governed by a dual-layer configuration design:

- **`settings.ini`**: Establishes the baseline training parameters, optimized by default for resource-constrained environments.
- **`train.py`**: Handles orchestration and parsing of runtime overrides.

*Example `settings.ini`:*
```ini
[DEFAULT]
DATASET=datasets/training/data.yaml
MODEL=yolo11n.pt
EPOCHS=10
BATCH=2
WORKERS=2
IMGSZ=416
```

## Contributing

We welcome structural improvements, architectural explorations, and optimizations. For detailed guidelines regarding bug reporting, feature proposals, and the pull request workflow, please review our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. Please refer to the **[LICENSE](LICENSE)** file for comprehensive details.
