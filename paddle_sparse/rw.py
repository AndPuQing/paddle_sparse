import paddle
from paddle_sparse.tensor import SparseTensor


def random_walk(
    src: SparseTensor, start: paddle.Tensor, walk_length: int
) -> paddle.Tensor:
    rowptr, col, _ = src.csr()
    return torch.ops.paddle_sparse.random_walk(rowptr, col, start, walk_length)


SparseTensor.random_walk = random_walk
