import master_thesis_code.decorators as decorators_module
from master_thesis_code.decorators import if_plotting_activated, timer_decorator


def test_if_plotting_activated_disabled(monkeypatch: object) -> None:
    """When IS_PLOTTING_ACTIVATED is False the decorated function must not run and must return None.

    The decorator reads the module-level IS_PLOTTING_ACTIVATED name at call time,
    so we patch it directly on the decorators module.
    """
    monkeypatch.setattr(decorators_module, "IS_PLOTTING_ACTIVATED", False)  # type: ignore[attr-defined]

    call_count = 0

    @if_plotting_activated
    def _plot() -> int:
        nonlocal call_count
        call_count += 1
        return 42

    result = _plot()
    assert result is None
    assert call_count == 0


def test_if_plotting_activated_enabled(monkeypatch: object) -> None:
    """When IS_PLOTTING_ACTIVATED is True the decorated function runs and its return value is passed through."""
    monkeypatch.setattr(decorators_module, "IS_PLOTTING_ACTIVATED", True)  # type: ignore[attr-defined]

    call_count = 0

    @if_plotting_activated
    def _plot() -> int:
        nonlocal call_count
        call_count += 1
        return 42

    result = _plot()
    assert result == 42
    assert call_count == 1


def test_timer_decorator_returns_value() -> None:
    """timer_decorator must pass through the wrapped function's return value."""

    @timer_decorator
    def _compute(x: int, y: int) -> int:
        return x + y

    result = _compute(3, 4)
    assert result == 7


def test_timer_decorator_preserves_name() -> None:
    """timer_decorator must preserve __name__ via @functools.wraps."""

    @timer_decorator
    def _my_function() -> None:
        pass

    assert _my_function.__name__ == "_my_function"


def test_timer_decorator_calls_function() -> None:
    """timer_decorator must actually call the wrapped function."""
    call_count = 0

    @timer_decorator
    def _side_effect() -> None:
        nonlocal call_count
        call_count += 1

    _side_effect()
    assert call_count == 1
