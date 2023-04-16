#pragma once

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
    CHECK_CPU(x);          \
    CHECK_CONTIGUOUS(x)

#define _SQR(x) ((x) * (x))

// from tiny-cuda-nn
/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                            \
    do {                                                               \
        cudaError_t result = x;                                        \
        if (result != cudaSuccess)                                     \
            throw std::runtime_error(                                  \
                std::string(FILE_LINE " " #x " failed with error: ") + \
                cudaGetErrorString(result));                           \
    } while (0)

template <typename T>
__host__ __device__ __inline__ T clamp(T val, T min, T max) {
    return fminf(fmaxf(val, min), max);
}

// Linear interp
// Subtract and fused multiply-add
// (1-w) a + w b
template <typename T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}
