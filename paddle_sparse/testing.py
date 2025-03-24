from __future__ import annotations
from typing import TYPE_CHECKING, Any

import paddle


if TYPE_CHECKING:
    from paddle._typing import (
        DTypeLike,
        PlaceLike,
    )

reductions = ["sum", "add", "mean", "min", "max"]


dtypes = [paddle.float32, paddle.float64, paddle.int32, paddle.int64]
grad_dtypes = [paddle.float32, paddle.float64]


places = ["cpu"]
if paddle.is_compiled_with_cuda():
    places += ["gpu:0"]


def tensor(x: Any, dtype: DTypeLike, place: PlaceLike):
    return None if x is None else paddle.to_tensor(x, dtype=dtype, place=place)
