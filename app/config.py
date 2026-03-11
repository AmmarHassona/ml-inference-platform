import os 
from pathlib import Path

# base directory of config file
BASE_DIR = Path(__file__).resolve().parent.parent

# define model path relative to the base directory
MODEL_PATH = BASE_DIR / "model_artifacts" / "model_v1.onnx"
SHADOE_MODEL_PATH = BASE_DIR / "model_artifacts" / "model_v2.onnx"