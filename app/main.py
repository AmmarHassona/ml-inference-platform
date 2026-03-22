from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel, field_validator
import onnxruntime as ort
import numpy as np
import time
import threading
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from app.config import MODEL_PATH, SHADOW_MODEL_PATH, MINI_LM_PATH, DRIFT_CHECK_INTERVAL
from app.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY, PREDICTION_PROBABILITY
from app.services.drift import record_features, run_drift_check, initialize_drift
from app.services.shadow import run_shadow_inference
from app.services.router import get_active_model, run_rollback_check
from app.services.embedding_drift import compute_embedding_drift, record_embedding
import app.services.topic_classification as topic_classification
from transformers import AutoTokenizer
from app.logger import setup_logging, get_logger

logger = get_logger("main")

def start_drift_scheduler():
    def loop():
        while True:
            time.sleep(DRIFT_CHECK_INTERVAL)
            try:
                run_drift_check()
                run_rollback_check()
                compute_embedding_drift()
            except Exception as e:
                logger.error("scheduler_error", error=str(e))
    thread = threading.Thread(target = loop, daemon = True)
    thread.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # load models once at startup
    app.state.session_v1 = ort.InferenceSession(str(MODEL_PATH))
    app.state.session_v2 = ort.InferenceSession(str(SHADOW_MODEL_PATH))
    app.state.minilm_session = ort.InferenceSession(str(MINI_LM_PATH / "model.onnx"))
    app.state.minilm_tokenizer = AutoTokenizer.from_pretrained(str(MINI_LM_PATH))
    topic_classification.load_corpus(app.state.minilm_session, app.state.minilm_tokenizer)
    initialize_drift()
    start_drift_scheduler()
    yield

app = FastAPI(lifespan = lifespan)

Instrumentator().instrument(app).expose(app)

class InferenceRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, v):
        if len(v) != 4:
            raise ValueError(f"Expected 4 features (sepal length, sepal width, petal length, petal width), got {len(v)}")
        return v

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request, body: InferenceRequest, background_tasks: BackgroundTasks):
    model_version = get_active_model()
    session = request.app.state.session_v1 if model_version == "v1" else request.app.state.session_v2
    features = np.array(body.features, dtype = np.float32).reshape(1, -1)
    record_features(body.features)
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: features})
    latency = time.perf_counter() - t0

    PREDICTION_LATENCY.labels(model_version = model_version).observe(latency)

    prediction = int(outputs[0][0])
    PREDICTION_COUNTER.labels(model_version = model_version).inc(1)
    prob_dict = outputs[1][0]
    probabilities = [float(prob_dict[i]) for i in sorted(prob_dict.keys())]
    PREDICTION_PROBABILITY.labels(model_version = model_version).observe(max(probabilities))

    background_tasks.add_task(run_shadow_inference, features, prediction, request.app.state)
    
    return {
        "prediction": prediction,
        "probabilities": probabilities
    }

@app.post("/predict/text")
async def predict_text(request: Request, body: TextRequest):
    session = request.app.state.minilm_session
    tokenizer = request.app.state.minilm_tokenizer
    sentences = body.text

    embedding = topic_classification.embed_text(sentences, session, tokenizer)
    label, confidence, matched_document = topic_classification.find_nearest(embedding)

    record_embedding(embedding)

    return {
        "label": label,
        "similarity": confidence,
        "matched_document": matched_document
    }