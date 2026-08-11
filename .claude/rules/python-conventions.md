---
paths: ["darksiren_emri/**/*.py", "darksiren_emri_test/**/*.py"]
description: Dataclass and typing conventions — mutable-default handling and mandatory type annotations
---

# Python Conventions

## Dataclass Conventions

Never use a mutable object as a bare default in a `@dataclass`. Python 3.13 raises `ValueError` at class-definition time. Always wrap with `field(default_factory=...)`:

```python
# Wrong: bar: MyMutableClass = MyMutableClass()  — crashes Python 3.13
# Correct:
bar: MyMutableClass = field(default_factory=MyMutableClass)
```

## Typing Conventions

All public and private functions/methods must have complete type annotations on every parameter and on the return type. The only exception is `__init__` where the return type may be omitted.

- Use `list[float]` not `List[float]`, `dict[str, int]` not `Dict[str, int]`, `X | None` not `Optional[X]`. Do **not** add `from __future__ import annotations`.
- Use `npt.NDArray[np.float64]` for typed arrays. Never use bare `np.ndarray` without a dtype parameter.
- CuPy has no mypy stubs. Annotate GPU-capable functions with `npt.NDArray[np.float64]` and add a comment that cupy arrays are also accepted at runtime. Never use `cp.ndarray` as a type annotation.
- Use `Callable` from `typing`, never lowercase `callable`. For signature-preserving decorators, use a `TypeVar` bound to `Callable[..., Any]` with `@functools.wraps`.

**mypy:** Config in `pyproject.toml`. Key flags: `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`. CuPy, `few`, `fastlisaresponse`, and `GPUtil` are under `ignore_missing_imports`.
