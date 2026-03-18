import argparse
import configparser
import os
import torch
from ultralytics import YOLO

def model_training(model: YOLO, dataset: str, output: str, epochs: int = 10, batch: int = 2, workers: int = 2, imgsz: int = 416):
  """
  Trains a YOLO model on the specified dataset.

  Args:
    model (YOLO): The YOLO model instance to train.
    dataset (str): Path to the dataset YAML configuration file.
    output (str): Directory where training results will be saved.
    epochs (int, optional): Number of training epochs. Defaults to 10.
    batch (int, optional): Batch size for training. Defaults to 2.
    workers (int, optional): Number of data loading workers. Defaults to 2.
    imgsz (int, optional): Target image size for training. Defaults to 416.

  Returns:
    results: The training results.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")
  model.to(device)
  
  results = model.train(
    data=dataset,
    epochs=epochs,
    batch=batch,
    workers=workers,
    imgsz=imgsz,
    device=device,
    project=output,
    cache=False,
    save=True
  )

  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train BarnSight Excrement Detection Model")
  parser.add_argument("--config", type=str, default="settings.ini", help="Path to configuration file")
  parser.add_argument("--model", type=str, help="Path to initial weights (e.g., yolo11n.pt)")
  parser.add_argument("--epochs", type=int, help="Override number of epochs")
  parser.add_argument("--batch", type=int, help="Override batch size")
  parser.add_argument("--imgsz", type=int, help="Override image size")
  
  args = parser.parse_args()

  # Load configures from config file
  config = configparser.ConfigParser()
  if not os.path.exists(args.config):
    print(f"Warning: Config file '{args.config}' not found. Using default settings if not provided via CLI.")
    config["DEFAULT"] = {
      "DATASET": "datasets/training/data.yaml",
      "MODEL": "yolo11n.pt",
      "EPOCHS": "10",
      "BATCH": "2",
      "WORKERS": "2",
      "IMGSZ": "416"
    }
  else:
    config.read(args.config)

  # Allow CLI arguments to override config file
  dataset = config["DEFAULT"].get("DATASET", "datasets/training/data.yaml")
  model_path = args.model if args.model else config["DEFAULT"].get("MODEL", "yolo11n.pt")
  epochs = args.epochs if args.epochs else int(config["DEFAULT"].get("EPOCHS", 10))
  batch = args.batch if args.batch else int(config["DEFAULT"].get("BATCH", 2))
  workers = int(config["DEFAULT"].get("WORKERS", 2))
  imgsz = args.imgsz if args.imgsz else int(config["DEFAULT"].get("IMGSZ", 416))

  # Load a model
  print(f"Loading model: {model_path}")
  model = YOLO(model_path) 

  # Ensure output directory exists
  output_dir = os.path.join(os.getcwd(), "datasets", "result")
  os.makedirs(output_dir, exist_ok=True)

  print(f"Starting training with dataset: {dataset}, epochs: {epochs}, batch: {batch}, workers: {workers}, imgsz: {imgsz}")

  # Train a model
  model_training(
    model=model,
    dataset=dataset,
    output=output_dir,
    epochs=epochs,
    batch=batch,
    workers=workers,
    imgsz=imgsz
  )
