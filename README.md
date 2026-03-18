# BarnSight - Model Training Pipeline

**Model training, experimentation, and optimization for on-device animal excrement detection.**

This repository contains the **training and experimentation pipeline** for the **BarnSight** computer vision models.
Its sole responsibility is to produce **efficient, accurate models** that can later be deployed to edge devices running inside barns.

If the edge device is the eyes, this repo is the school they went to.

## 🎯 Purpose

Farm environments are visually chaotic:

- uneven lighting
- dirty floors
- occlusions
- different animal species
- wildly inconsistent “ground truth”

This repository exists to turn that chaos into robust, production-ready models that can reliably detect visible animal excrement under real farm conditions.

The output of this repo is trained model artifacts (e.g., `.pt`, `.onnx`), not a running service.

## 🚀 Getting Started

### Prerequisites

You need [uv](https://github.com/astral-sh/uv) installed to manage the Python environment and dependencies efficiently.

```bash
# Clone the repository
git clone https://github.com/barnsight/train.git barnsight-train
cd barnsight-train

# Install dependencies using uv
uv sync
```

### Dataset Structure

The training pipeline expects a YOLO-formatted dataset. By default, it looks for `datasets/training/data.yaml`. Your dataset should look like this:

```
datasets/training/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## 🧪 Training Workflow

1. **Prepare and label farm images**
    - Capture variations: Different animals, floor materials, lighting conditions, and contamination levels.
2. **Configure training parameters**
    - Modify `settings.ini` to adjust basic training configurations.
3. **Train the model**
    - Run the training script locally or use the provided Colab notebook for GPU access.
4. **Export trained weights**
    - The output models will be saved in `datasets/result/train/weights/`.
    - These `.pt` files are ready to be optimized and placed in the edge device `models/` directory.

### Running Training Locally

You can start training using the default `settings.ini`:

```bash
uv run train.py
```

Override settings using command-line arguments:

```bash
uv run train.py --epochs 50 --batch 16 --model yolo11s.pt
```

## ⚙️ Configuration

Training behavior is controlled via:

- **`settings.ini`** — Default model and dataset configuration.
- **`train.py`** — Training logic, orchestration, and CLI arguments.

Example `settings.ini`:
```ini
[DEFAULT]
DATASET=datasets/training/data.yaml
EPOCHS=10
BATCH=4
WORKERS=8
```

## 🤝 Contributing

We welcome contributions! Whether it's tweaking the augmentation strategy, proposing a better base model, or improving the data loading pipeline.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and our coding standards.

## 📄 License

Licensed under the MIT License. See the **LICENSE** file for details.
