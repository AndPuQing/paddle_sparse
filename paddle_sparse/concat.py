from typing import Optional, List, Tuple  # noqa

import paddle
from paddle_sparse.storage import SparseStorage
from paddle_sparse.tensor import SparseTensor


# noqa: F811
def concat(tensors, axis):  # noqa: F811
    # type: (List[SparseTensor], int) -> SparseTensor
    pass


# noqa: F811
def concat(tensors, axis):  # noqa: F811
    # type: (List[SparseTensor], Tuple[int, int]) -> SparseTensor
    pass


# noqa: F811
def concat(tensors, axis):  # noqa: F811
    # type: (List[SparseTensor], List[int]) -> SparseTensor
    pass


def concat(tensors, axis):  # noqa: F811
    assert len(tensors) > 0

    if isinstance(axis, int):
        axis = tensors[0].dim() + axis if axis < 0 else axis

        if axis == 0:
            return concat_first(tensors)
        elif axis == 1:
            return concat_second(tensors)
        elif axis > 1 and axis < tensors[0].dim():
            values = []
            for tensor in tensors:
                value = tensor.storage.value()
                assert value is not None
                values.append(value)
            value = paddle.concat(values, axis=axis - 1)
            return tensors[0].set_value(value, layout="coo")

        else:
            raise IndexError(
                (
                    f"Dimension out of range: Expected to be in range of "
                    f"[{-tensors[0].dim()}, {tensors[0].dim() - 1}], but got "
                    f"{axis}."
                )
            )
    else:
        assert isinstance(axis, (tuple, list))
        assert len(axis) == 2
        assert sorted(axis) == [0, 1]
        return concat_diag(tensors)


def concat_first(tensors: List[SparseTensor]) -> SparseTensor:
    rows: List[paddle.Tensor] = []
    rowptrs: List[paddle.Tensor] = []
    cols: List[paddle.Tensor] = []
    values: List[paddle.Tensor] = []
    sparse_sizes: List[int] = [0, 0]
    rowcounts: List[paddle.Tensor] = []

    nnz: int = 0
    for tensor in tensors:
        row = tensor.storage._row
        if row is not None:
            rows.append(row + sparse_sizes[0])

        rowptr = tensor.storage._rowptr
        if rowptr is not None:
            rowptrs.append(rowptr[1:] + nnz if len(rowptrs) > 0 else rowptr)

        cols.append(tensor.storage._col)

        value = tensor.storage._value
        if value is not None:
            values.append(value)

        rowcount = tensor.storage._rowcount
        if rowcount is not None:
            rowcounts.append(rowcount)

        sparse_sizes[0] += tensor.sparse_size(0)
        sparse_sizes[1] = max(sparse_sizes[1], tensor.sparse_size(1))
        nnz += tensor.nnz()

    row: Optional[paddle.Tensor] = None
    if len(rows) == len(tensors):
        row = paddle.concat(rows, axis=0)

    rowptr: Optional[paddle.Tensor] = None
    if len(rowptrs) == len(tensors):
        rowptr = paddle.concat(rowptrs, axis=0)

    col = paddle.concat(cols, axis=0)

    value: Optional[paddle.Tensor] = None
    if len(values) == len(tensors):
        value = paddle.concat(values, axis=0)

    rowcount: Optional[paddle.Tensor] = None
    if len(rowcounts) == len(tensors):
        rowcount = paddle.concat(rowcounts, axis=0)

    storage = SparseStorage(
        row=row,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=(sparse_sizes[0], sparse_sizes[1]),
        rowcount=rowcount,
        colptr=None,
        colcount=None,
        csr2csc=None,
        csc2csr=None,
        is_sorted=True,
    )
    return tensors[0].from_storage(storage)


def concat_second(tensors: List[SparseTensor]) -> SparseTensor:
    rows: List[paddle.Tensor] = []
    cols: List[paddle.Tensor] = []
    values: List[paddle.Tensor] = []
    sparse_sizes: List[int] = [0, 0]
    colptrs: List[paddle.Tensor] = []
    colcounts: List[paddle.Tensor] = []

    nnz: int = 0
    for tensor in tensors:
        row, col, value = tensor.coo()
        rows.append(row)
        cols.append(tensor.storage._col + sparse_sizes[1])

        if value is not None:
            values.append(value)

        colptr = tensor.storage._colptr
        if colptr is not None:
            colptrs.append(colptr[1:] + nnz if len(colptrs) > 0 else colptr)

        colcount = tensor.storage._colcount
        if colcount is not None:
            colcounts.append(colcount)

        sparse_sizes[0] = max(sparse_sizes[0], tensor.sparse_size(0))
        sparse_sizes[1] += tensor.sparse_size(1)
        nnz += tensor.nnz()

    row = paddle.concat(rows, axis=0)
    col = paddle.concat(cols, axis=0)

    value: Optional[paddle.Tensor] = None
    if len(values) == len(tensors):
        value = paddle.concat(values, axis=0)

    colptr: Optional[paddle.Tensor] = None
    if len(colptrs) == len(tensors):
        colptr = paddle.concat(colptrs, axis=0)

    colcount: Optional[paddle.Tensor] = None
    if len(colcounts) == len(tensors):
        colcount = paddle.concat(colcounts, axis=0)

    storage = SparseStorage(
        row=row,
        rowptr=None,
        col=col,
        value=value,
        sparse_sizes=(sparse_sizes[0], sparse_sizes[1]),
        rowcount=None,
        colptr=colptr,
        colcount=colcount,
        csr2csc=None,
        csc2csr=None,
        is_sorted=False,
    )
    return tensors[0].from_storage(storage)


def concat_diag(tensors: List[SparseTensor]) -> SparseTensor:
    assert len(tensors) > 0

    rows: List[paddle.Tensor] = []
    rowptrs: List[paddle.Tensor] = []
    cols: List[paddle.Tensor] = []
    values: List[paddle.Tensor] = []
    sparse_sizes: List[int] = [0, 0]
    rowcounts: List[paddle.Tensor] = []
    colptrs: List[paddle.Tensor] = []
    colcounts: List[paddle.Tensor] = []
    csr2cscs: List[paddle.Tensor] = []
    csc2csrs: List[paddle.Tensor] = []

    nnz: int = 0
    for tensor in tensors:
        row = tensor.storage._row
        if row is not None:
            rows.append(row + sparse_sizes[0])

        rowptr = tensor.storage._rowptr
        if rowptr is not None:
            rowptrs.append(rowptr[1:] + nnz if len(rowptrs) > 0 else rowptr)

        cols.append(tensor.storage._col + sparse_sizes[1])

        value = tensor.storage._value
        if value is not None:
            values.append(value)

        rowcount = tensor.storage._rowcount
        if rowcount is not None:
            rowcounts.append(rowcount)

        colptr = tensor.storage._colptr
        if colptr is not None:
            colptrs.append(colptr[1:] + nnz if len(colptrs) > 0 else colptr)

        colcount = tensor.storage._colcount
        if colcount is not None:
            colcounts.append(colcount)

        csr2csc = tensor.storage._csr2csc
        if csr2csc is not None:
            csr2cscs.append(csr2csc + nnz)

        csc2csr = tensor.storage._csc2csr
        if csc2csr is not None:
            csc2csrs.append(csc2csr + nnz)

        sparse_sizes[0] += tensor.sparse_size(0)
        sparse_sizes[1] += tensor.sparse_size(1)
        nnz += tensor.nnz()

    row: Optional[paddle.Tensor] = None
    if len(rows) == len(tensors):
        row = paddle.concat(rows, axis=0)

    rowptr: Optional[paddle.Tensor] = None
    if len(rowptrs) == len(tensors):
        rowptr = paddle.concat(rowptrs, axis=0)

    col = paddle.concat(cols, axis=0)

    value: Optional[paddle.Tensor] = None
    if len(values) == len(tensors):
        value = paddle.concat(values, axis=0)

    rowcount: Optional[paddle.Tensor] = None
    if len(rowcounts) == len(tensors):
        rowcount = paddle.concat(rowcounts, axis=0)

    colptr: Optional[paddle.Tensor] = None
    if len(colptrs) == len(tensors):
        colptr = paddle.concat(colptrs, axis=0)

    colcount: Optional[paddle.Tensor] = None
    if len(colcounts) == len(tensors):
        colcount = paddle.concat(colcounts, axis=0)

    csr2csc: Optional[paddle.Tensor] = None
    if len(csr2cscs) == len(tensors):
        csr2csc = paddle.concat(csr2cscs, axis=0)

    csc2csr: Optional[paddle.Tensor] = None
    if len(csc2csrs) == len(tensors):
        csc2csr = paddle.concat(csc2csrs, axis=0)

    storage = SparseStorage(
        row=row,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=(sparse_sizes[0], sparse_sizes[1]),
        rowcount=rowcount,
        colptr=colptr,
        colcount=colcount,
        csr2csc=csr2csc,
        csc2csr=csc2csr,
        is_sorted=True,
    )
    return tensors[0].from_storage(storage)
