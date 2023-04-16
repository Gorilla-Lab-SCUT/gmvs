#pragma once

#include <assert.h>
#include <iostream>
#include <tuple>
#include <vector>

#define MAX_NGB 32

using std::tuple;
using std::vector;

struct PatchMatchParams {
    int32_t max_iterations = 3;
    int32_t patch_size = 11;
    int32_t num_images = 5;
    int32_t max_image_size = 3200;
    int32_t radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int32_t top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;
    bool geom_consistency = false;
    bool multi_geometry = false;
    bool planar_prior = false;
    bool superpixel_filter = false;
};

struct Problem {
    int32_t num_ngb;
    int32_t ref_image_id;
    int32_t src_image_ids[MAX_NGB];
};

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int32_t height;
    int32_t width;
    float depth_min;
    float depth_max;
};

Camera load_camera(vector<float>, vector<float>, vector<float>, int32_t, int32_t, float, float);

Problem load_problem(int32_t, vector<int32_t>);
