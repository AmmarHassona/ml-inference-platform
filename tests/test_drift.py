import numpy as np
import pytest
from app.services.drift import calculate_psi, initialize_drift
import app.services.drift as drift # initialize_drift runs first and reference_features is not None

@pytest.fixture(autouse=True)
def reset_drift_state():
    drift.reference_features = None
    drift.bin_edges_per_feature.clear()
    drift.ref_pcts.clear()
    drift.feature_window.clear()

def test_psi_calculation():
    initialize_drift()

    result = calculate_psi(drift.reference_features)

    for score in result.values():
        assert score < 0.1

def test_psi_high_on_out_of_distribution_features():
    initialize_drift()
    
    ood_features = np.array([[100.0, 200.0, 300.0, 400.0]] * 50)  # shape (50, 4)
    result = calculate_psi(ood_features)
    
    for score in result.values():
        assert score > 0.2