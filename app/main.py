from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import time
from app.config import MODEL_PATH
from app.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY, PREDICTION_PROBABILITY

from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once at startup
    app.state.session = ort.InferenceSession(str(MODEL_PATH))
    yield

app = FastAPI(lifespan = lifespan)

Instrumentator().instrument(app).expose(app)

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
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: features})
    latency = time.perf_counter() - t0
    PREDICTION_LATENCY.labels(model_version = "v1").observe(latency)

    prediction = int(outputs[0][0])
    PREDICTION_COUNTER.labels(model_version = "v1").inc(1)
    prob_dict = outputs[1][0]
    probabilities = [float(prob_dict[i]) for i in sorted(prob_dict.keys())]
    PREDICTION_PROBABILITY.labels(model_version = "v1").observe(max(probabilities))
    
    return {
        "prediction": prediction,
        "probabilities": probabilities
    }