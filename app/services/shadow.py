import numpy as np
from collections import deque
from app.metrics import SHADOW_DIVERGENCE
from app.services.router import update_divergence

# rolling buffer of (v1_prediction, v2_prediction) pairs
_comparison_buffer = deque(maxlen = 200)

async def run_shadow_inference(features: np.ndarray, v1_prediction: int, app_state):
    try:
        session_v2 = app_state.session_v2
        input_name = session_v2.get_inputs()[0].name
        outputs = session_v2.run(None, {input_name: features})
        v2_prediction = int(outputs[0][0])
        _comparison_buffer.append((v1_prediction, v2_prediction))
        # 5. if buffer has enough data (>=20), compute divergence rate
        # divergence = percentage of recent requests where v1 and v2 disagreed
        if len(_comparison_buffer) >= 20:
            divergence = sum(v1 != v2 for v1, v2 in _comparison_buffer) / len(_comparison_buffer)
            SHADOW_DIVERGENCE.set(divergence)
        update_divergence(divergence)
    except Exception as e:
        print(f"Shadow inference failed: {e}", flush = True)