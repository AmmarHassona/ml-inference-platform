import random
from app.metrics import ROLLBACK_COUNTER

_current_divergence = 0.0
active_model = "v1"
canary_percent = 10
DIVERGENCE_THRESHOLD = 0.15

def get_active_model() -> str:
    if random.random() * 100 < canary_percent:
        return "v2"
    else:
        return "v1"
    
def trigger_rollback():
    global canary_percent
    canary_percent = 0
    print("ROLLBACK TRIGGERED: routing all traffic to v1", flush = True)
    ROLLBACK_COUNTER.inc(1)

def update_divergence(value: float):
    global _current_divergence
    _current_divergence = value

def run_rollback_check():
    if _current_divergence > DIVERGENCE_THRESHOLD:
        trigger_rollback()