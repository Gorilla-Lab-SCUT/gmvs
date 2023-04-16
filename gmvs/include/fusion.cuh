// Copyright 2022 Gorilla-Lab
#pragma once

#include <iomanip>

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include "data_structure.hpp"

namespace mvs {
class Fuser {
public:
    Fuser(){};
    ~Fuser(){};
    void load_samples(
        const vector<Problem> problems,
        const vector<Camera> cameras,
        const Tensor& depths,
        const Tensor& normals,
        const Tensor& masks);
    tuple<Tensor, Tensor> run_fusion(
        const int32_t problem_idx,
        const int32_t geom_consistent);

    Tensor depths_host;
    Tensor normals_host;
    Tensor masks_cuda;
private:
    int32_t height;
    int32_t width;
    // host
    vector<Camera> cameras_host;
    vector<Problem> problems_host;
};
}  // namespace mvs
