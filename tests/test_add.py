from itertools import product

import pytest
import paddle

from paddle_sparse import SparseTensor, add
from paddle_sparse.testing import places, dtypes, tensor


@pytest.mark.parametrize("dtype,place", product(dtypes, places))
def test_add(dtype, place):
    rowA = paddle.to_tensor([0, 0, 1, 2, 2], place=place)
    colA = paddle.to_tensor([0, 2, 1, 0, 1], place=place)
    valueA = tensor([1, 2, 4, 1, 3], dtype, place)
    A = SparseTensor(row=rowA, col=colA, value=valueA)

    rowB = paddle.to_tensor([0, 0, 1, 2, 2], place=place)
    colB = paddle.to_tensor([1, 2, 2, 1, 2], place=place)
    valueB = tensor([2, 3, 1, 2, 4], dtype, place)
    B = SparseTensor(row=rowB, col=colB, value=valueB)

    C = A + B
    rowC, colC, valueC = C.coo()

    assert rowC.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert colC.tolist() == [0, 1, 2, 1, 2, 0, 1, 2]
    assert valueC.tolist() == [1, 2, 5, 4, 1, 1, 5, 4]

    add(A, B)
