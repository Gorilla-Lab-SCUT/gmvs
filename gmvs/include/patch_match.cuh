// Copyright 2022 Gorilla-Lab
#pragma once

#include <iomanip>

#include <torch/extension.h>
// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include "data_structure.hpp"

using mvs::Camera;
using mvs::PatchMatchParams;
using mvs::Problem;

namespace mvs {
class PatchMatcher {
public:
    PatchMatcher();
    ~PatchMatcher();
    void add_samples(
        const vector<Problem> problems,
        const vector<Camera> cameras,
        const Tensor& images,
        const Tensor& depths,
        const Tensor& normals,
        const Tensor& costs);

    tuple<Tensor, Tensor, Tensor> run_patch_match(const int32_t problem_idx, const bool verbose);

    PatchMatchParams params;

private:
    Tensor images_host;
    Tensor depths_host;
    Tensor normals_host;
    Tensor costs_host;
    int32_t height;
    int32_t width;
    int32_t num_images;
    // host
    vector<Camera> cameras_host;
    vector<Problem> problems_host;
    // cuda
    // Camera* cameras_cuda;
    // Problem* problems_cuda;
};
}  // namespace mvs
