from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from app.config import MINI_LM_PATH

model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export = True
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model.save_pretrained(str(MINI_LM_PATH))
tokenizer.save_pretrained(str(MINI_LM_PATH))