from master_thesis_code.decorators import timer_decorator


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
