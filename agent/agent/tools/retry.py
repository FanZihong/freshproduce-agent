"""重试与超时包装。

工具失败时按指数退避重试；重试耗尽抛出 ToolExecutionError，
由网关翻译成降级动作。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ToolExecutionError(RuntimeError):
    """重试耗尽后抛出，携带尝试次数与最后一次异常。"""

    def __init__(self, message: str, attempts: int, last_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryOutcome:
    value: Any
    attempts: int


def run_with_retry(
    func: Callable[..., T],
    *args: Any,
    attempts: int = 3,
    base_delay: float = 0.2,
    backoff: float = 2.0,
    timeout: float | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> RetryOutcome:
    """执行 func 并在失败时重试，返回结果与真实尝试次数。"""
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            if timeout is None:
                return RetryOutcome(func(*args, **kwargs), attempt)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, *args, **kwargs)
                return RetryOutcome(future.result(timeout=timeout), attempt)
        except FutureTimeoutError as exc:
            last_error = exc
        except retry_on as exc:
            last_error = exc

        if attempt < attempts:
            time.sleep(base_delay * (backoff ** (attempt - 1)))

    raise ToolExecutionError(
        f"{getattr(func, '__name__', 'tool')} 连续 {attempts} 次调用失败：{last_error}",
        attempts=attempts,
        last_error=last_error,
    )


def with_retry(func=None, **options):
    """装饰器形式：@with_retry(attempts=3)，返回原始结果（不带 attempts）。"""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            return run_with_retry(fn, *args, **options, **kwargs).value

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
