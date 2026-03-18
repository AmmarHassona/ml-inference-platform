from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MINI_LM_PATH = BASE_DIR / "model_artifacts" / "minilm"
MINI_LM_PATH.mkdir(parents=True, exist_ok=True)

model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True
)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model.save_pretrained(str(MINI_LM_PATH))
tokenizer.save_pretrained(str(MINI_LM_PATH))
print("Done")