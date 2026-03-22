import random
from app.metrics import ROLLBACK_COUNTER
from app.config import CANARY_PERCENT, DIVERGENCE_THRESHOLD
from app.logger import get_logger
import threading

logger = get_logger("router")

_divergence_lock = threading.Lock()
_current_divergence = 0.0
canary_percent = CANARY_PERCENT

def get_active_model() -> str:
    if random.random() * 100 < canary_percent:
        return "v2"
    else:
        return "v1"
    
def trigger_rollback():
    global canary_percent
    prev = canary_percent
    canary_percent = 0
    logger.warning("rollback_triggered", divergence=_current_divergence, canary_percent_before=prev, canary_percent_after=0)
    ROLLBACK_COUNTER.inc(1)

def update_divergence(value: float):
    global _current_divergence
    with _divergence_lock:
        _current_divergence = value

def restore_canary():
    global canary_percent
    canary_percent = CANARY_PERCENT
    logger.info("canary_restored", divergence=_current_divergence, canary_percent_after=CANARY_PERCENT)

def run_rollback_check():
    with _divergence_lock:
        divergence = _current_divergence
    if divergence > DIVERGENCE_THRESHOLD:
        if canary_percent > 0:
            trigger_rollback()
    elif canary_percent == 0:
        restore_canary()