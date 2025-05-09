#pragma once

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")

// __device__ __inline__ at::Half __shfl_up_sync(const unsigned mask,
//                                               const at::Half var,
//                                               const unsigned int delta) {
//   return __shfl_up_sync(mask, var.operator __half(), delta);
// }

// __device__ __inline__ at::Half __shfl_down_sync(const unsigned mask,
//                                                 const at::Half var,
//                                                 const unsigned int delta) {
//   return __shfl_down_sync(mask, var.operator __half(), delta);
// }

// __device__ __inline__ at::Half __shfl_sync(const unsigned mask,
//                                            const at::Half var,
//                                            const int delta) {
//   return __shfl_sync(mask, var.operator __half(), delta);
// }

// __device__ __inline__ at::Half __shfl_up(const at::Half var,
//                                          const unsigned int delta) {
//   return __shfl_up(var.operator __half(), delta);
// }

// __device__ __inline__ at::Half __shfl_down(const at::Half var,
//                                            const unsigned int delta) {
//   return __shfl_down(var.operator __half(), delta);
// }

// __device__ __inline__ at::Half
// __shfl(const at::Half var, const int delta) {
//   return __shfl(var.operator __half(), delta);
// }

// #ifdef USE_ROCM
// __device__ __inline__ at::Half __ldg(const at::Half* ptr) {
//   return __ldg(reinterpret_cast<const __half*>(ptr));
// }
// #define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)
// #define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)
// #define SHFL_SYNC(mask, var, delta) __shfl(var, delta)
// #else
// #define SHFL_UP_SYNC __shfl_up_sync
// #define SHFL_DOWN_SYNC __shfl_down_sync
// #define SHFL_SYNC __shfl_sync
// #endif