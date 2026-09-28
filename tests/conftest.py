import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "metal: requires a native Apple Metal device")
    config.addinivalue_line("markers", "model: requires the selected model artifact")


def pytest_collection_modifyitems(items):
    metal = [item for item in items if item.get_closest_marker("metal")]
    if not metal:
        return
    try:
        import mlx.core as mx
        available = mx.metal.is_available()
    except (ImportError, RuntimeError, OSError):
        available = False
    if not available:
        for item in metal:
            item.add_marker(pytest.mark.skip(reason="native Metal device unavailable"))
