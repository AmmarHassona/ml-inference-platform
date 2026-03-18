from pathlib import Path

# Drift detection
PSI_WINDOW_SIZE: int = 500
PSI_MIN_SAMPLES: int = 50
PSI_N_BINS: int = 5

# Shadow mode
SHADOW_BUFFER_SIZE: int = 200
SHADOW_MIN_SAMPLES: int = 20

# Canary
CANARY_PERCENT: float = 10.0
DIVERGENCE_THRESHOLD: float = 0.15

# Embedding drift
EMBEDDING_REFERENCE_SIZE: int = 50
EMBEDDING_BUFFER_SIZE: int = 200
EMBEDDING_MIN_SAMPLES: int = 10

# Scheduler
DRIFT_CHECK_INTERVAL: int = 60

# base directory of config file
BASE_DIR = Path(__file__).resolve().parent.parent

# define model path relative to the base directory
MODEL_PATH = BASE_DIR / "model_artifacts" / "model_v1.onnx"
SHADOW_MODEL_PATH = BASE_DIR / "model_artifacts" / "model_v2.onnx"
REFERENCE_FEATURES_PATH = BASE_DIR / "model_artifacts" / "reference_features.npy"
MINI_LM_PATH = BASE_DIR / "model_artifacts" / "minilm"