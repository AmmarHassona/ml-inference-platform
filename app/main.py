from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

from contextlib import asynccontextmanager

from app.config import MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once at startup
    app.state.session = ort.InferenceSession(str(MODEL_PATH))
    yield

app = FastAPI(lifespan = lifespan)

class InferenceRequest(BaseModel):
    features:  list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: Request, body: InferenceRequest):
    session = request.app.state.session
    features = np.array(body.features, dtype = np.float32).reshape(1, -1)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    prediction = int(outputs[0][0])
    prob_dict = outputs[1][0]
    probabilities = [float(prob_dict[i]) for i in sorted(prob_dict.keys())]
    
    return {
        "prediction": prediction,
        "probabilities": probabilities
    }