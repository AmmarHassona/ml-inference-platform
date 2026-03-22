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

def test_rollback_does_not_fire_twice():
    router.canary_percent = 0
    router._current_divergence = 0.16
    before = router.ROLLBACK_COUNTER._value.get()
    router.run_rollback_check()

    assert router.ROLLBACK_COUNTER._value.get() == before

def test_canary_restored_when_divergence_normalises():
    router.canary_percent = 0
    router._current_divergence = 0.05
    router.run_rollback_check()

    assert router.canary_percent == router.CANARY_PERCENT