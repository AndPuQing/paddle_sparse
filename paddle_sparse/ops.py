import paddle_sparse_ops
import paddle


def custom_ptr2ind(x, e):
    if e == 0:
        return paddle.empty([0], dtype=x.dtype).to(x.place)
    else:
        return paddle_sparse_ops.custom_ptr2ind(x, e)


def custom_ind2ptr(x, m):
    return paddle_sparse_ops.custom_ind2ptr(x, m)
