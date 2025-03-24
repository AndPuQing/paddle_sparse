import pytest
import paddle

from paddle_sparse.concat import concat
from paddle_sparse.tensor import SparseTensor
from paddle_sparse.testing import places, tensor


@pytest.mark.parametrize("place", places)
def test_cat(place):
    row, col = tensor([[0, 0, 1], [0, 1, 2]], paddle.int64, place)
    mat1 = SparseTensor(row=row, col=col)
    mat1.fill_cache_()

    row, col = tensor([[0, 0, 1, 2], [0, 1, 1, 0]], paddle.int64, place)
    mat2 = SparseTensor(row=row, col=col)
    mat2.fill_cache_()

    out = concat([mat1, mat2], axis=0)
    assert out.to_dense().tolist() == [
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.has_rowcount()
    assert out.storage.num_cached_keys() == 1

    out = concat([mat1, mat2], axis=1)
    assert out.to_dense().tolist() == [
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ]
    assert out.storage.has_row()
    assert not out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 2

    out = concat([mat1, mat2], axis=(0, 1))
    assert out.to_dense().tolist() == [
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5

    value = paddle.randn((mat1.nnz(), 4)).to(place)
    mat1 = mat1.set_value_(value, layout="coo")
    out = concat([mat1, mat1], axis=-1)
    assert out.storage.value().shape == [mat1.nnz(), 8]
    assert out.storage.has_row()
    assert out.storage.has_rowptr()
    assert out.storage.num_cached_keys() == 5
