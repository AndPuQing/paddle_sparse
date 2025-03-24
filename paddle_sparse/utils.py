from typing import Any, Optional, Tuple

import paddle

import paddle_sparse.typing

try:
    from typing_extensions import Final  # noqa
except ImportError:
    pass


def index_sort(inputs: paddle.Tensor, max_value: Optional[int] = None) -> paddle.Tensor:
    r"""See pyg-lib documentation for more details:
    https://pyg-lib.readthedocs.io/en/latest/modules/ops.html"""
    return inputs.argsort()


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)
