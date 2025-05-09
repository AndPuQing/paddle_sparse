#include "paddle/extension.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/ext/tensor_compat.h"
#include "utils.h"
#include <cstdint>

std::vector<paddle::Tensor> ind2ptr_cpu_forward(const paddle::Tensor &ind,
                                                int64_t M) {
  CHECK_CPU(ind);
  paddle::Tensor out;
  auto ind_data = ind.data<int64_t>();
  int64_t numel = ind.numel();

  if (numel == 0) {
    out = paddle::zeros({M + 1}, ind.dtype());
    return {out};
  } else {
    out = paddle::empty({M + 1}, ind.dtype());
  }
  auto out_data = out.data<int64_t>();

  for (int64_t i = 0; i <= ind_data[0]; i++)
    out_data[i] = 0;

  int64_t idx = ind_data[0], next_idx;
  for (int64_t i = 0; i < numel - 1; i++) {
    next_idx = ind_data[i + 1];
    for (; idx < next_idx; idx++) {
      out_data[idx + 1] = i + 1;
    }
  }

  for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
    out_data[i] = numel;

  return {out};
}

std::vector<paddle::Tensor> ptr2ind_cpu_forward(const paddle::Tensor &ptr,
                                                int64_t E) {
  CHECK_CPU(ptr);
  paddle::Tensor out;
  auto ptr_data = ptr.data<int64_t>();
  if (E == 0) {
    out = paddle::empty({0}, ptr.dtype());
    return {out};
  } else {
    out = paddle::empty({E}, ptr.dtype());
  }

  auto out_data = out.data<int64_t>();

  int64_t numel = ptr.numel();

  int64_t idx = ptr_data[0], next_idx;
  for (int64_t i = 0; i < numel - 1; i++) {
    next_idx = ptr_data[i + 1];
    for (int64_t e = idx; e < next_idx; e++) {
      out_data[e] = i;
    }
    idx = next_idx;
  }

  return {out};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> ind2ptr_cuda_forward(const paddle::Tensor &ind,
                                                 int64_t M);
std::vector<paddle::Tensor> ptr2ind_cuda_forward(const paddle::Tensor &ptr,
                                                 int64_t E);
#endif

std::vector<paddle::Tensor> ind2ptrForward(const paddle::Tensor &ind,
                                           int64_t M) {
  if (ind.is_cpu()) {
    return ind2ptr_cpu_forward(ind, M);
#ifdef PADDLE_WITH_CUDA
  } else if (ind.is_gpu()) {
    return ind2ptr_cuda_forward(ind, M);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom "
             "gather_coo operator.");
  }
}

std::vector<paddle::Tensor> ptr2indForward(const paddle::Tensor &ptr,
                                           int64_t E) {
  if (ptr.is_gpu()) {
#ifdef PADDLE_WITH_CUDA
    return ptr2ind_cuda_forward(ptr, E);
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return ptr2ind_cpu_forward(ptr, E);
  }
}

std::vector<std::vector<int64_t>>
Ind2PtrInferShape(std::vector<int64_t> x_shape, int64_t M) {
  return {{M + 1}};
}

std::vector<paddle::DataType> Ind2PtrInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>>
Ptr2IndInferShape(std::vector<int64_t> x_shape, int64_t E) {
  return {{E}};
}

std::vector<paddle::DataType> Ptr2IndInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_ind2ptr)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"M: int64_t"})
    .SetKernelFn(PD_KERNEL(ind2ptrForward))
    .SetInferShapeFn(PD_INFER_SHAPE(Ind2PtrInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Ind2PtrInferDtype));

PD_BUILD_OP(custom_ptr2ind)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"E: int64_t"})
    .SetKernelFn(PD_KERNEL(ptr2indForward))
    .SetInferShapeFn(PD_INFER_SHAPE(Ptr2IndInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Ptr2IndInferDtype));