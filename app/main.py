from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI()

# load model once at startup
session = ort.InferenceSession("model_artifacts/model_v1.onnx")

class InferenceRequest(BaseModel):
    features:  list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: InferenceRequest):
    features = np.array(request.features, dtype = np.float32).reshape(1, -1)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    prediction = int(outputs[0][0])
    prob_dict = outputs[1][0]
    probabilities = [float(prob_dict[i]) for i in sorted(prob_dict.keys())]
    
    return {
        "prediction": prediction,
        "probabilities": probabilities
    }