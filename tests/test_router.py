import pytest
import app.services.router as router

@pytest.fixture(autouse=True)
def reset_router_states():
    router.canary_percent = router.CANARY_PERCENT
    router._current_divergence = 0

def test_rollback_fires_above_threshold():
    router._current_divergence = 0.16
    router.run_rollback_check()
    
    assert router.canary_percent == 0

def test_rollback_does_not_fire_at_threshold():
    router._current_divergence = 0.15
    router.run_rollback_check()

    assert router.canary_percent == 10.0