#pragma once

#include <iomanip>
// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

// opencv
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "planar_initer.hpp"

#define MAX_IMAGES 1024

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

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle(const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3)
        : pt1(_pt1), pt2(_pt2), pt3(_pt3) {}
};

cv::Mat sliding_window_normal(
    const cv::Mat&,
    const cv::Mat&,
    const int32_t);

cv::Mat fusion_planar_mask(
    const cv::Mat&,
    const cv::Mat&,
    const cv::Mat&,
    const int32_t,
    const int32_t,
    const int32_t,
    const float,
    const float,
    const float);

cv::Mat filter_by_var_map(
    const cv::Mat&,
    const cv::Mat&,
    const float);

class PlanarInit {
   public:
    PlanarInit(const bool);
    ~PlanarInit();

    // set/get images/depth
    void add_image(const std::string);
    void add_superpixel(const cv::Mat&);
    cv::Mat get_image(const int32_t);
    int32_t get_num_images();

    // set/get camera
    void add_camera(vector<float>, vector<float>, vector<float>, int32_t, int32_t, float, float);
    Camera get_camera(const int32_t);

    void cuda_space_init();
    void cuda_planar_prior_init();
    void run_patch_match();
    void jump_patch_match();

    // outputs
    void export_plane(cv::Mat&, cv::Mat&, cv::Mat&);
    float get_cost(const int32_t);

    vector<cv::Point> get_support_points();
    void vis_triangulation(cv::Rect, vector<Triangle>, const std::string);
    void triangulation(const cv::Mat&, const std::string);
    void process_mask(const cv::Mat&, const cv::Rect, vector<Triangle>);
    float3 get_3D_point_on_ref_cam(const int32_t, const int32_t, const float, const Camera);
    float4 get_prior_plane_params(const Triangle, const cv::Mat);
    float get_depth_from_plane_param(const float4, const int, const int);

    // member
    PatchMatchParams params;

   private:
    bool verbose;
    int num_images;
    cv::Mat superpixel;
    vector<cv::Mat> images;
    vector<cv::Mat> depths;
    vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4* plane_hypotheses_host;
    float* costs_host;
    float4* prior_planes_host;
    unsigned int* plane_masks_host;

    // plane parameters
    vector<float4> planeParams_tri;
    cv::Mat mask_tri;

    Camera* cameras_cuda;
    cudaArray* cuArray[MAX_IMAGES];
    cudaArray* cuDepthArray[MAX_IMAGES];
    cudaTextureObjects* texture_objects_cuda;
    cudaTextureObjects* texture_depths_cuda;
    float4* plane_hypotheses_cuda;
    float* costs_cuda;
    curandState* rand_states_cuda;
    unsigned int* selected_views_cuda;
    float* depths_cuda;
    float4* prior_planes_cuda;
    unsigned int* plane_masks_cuda;
};

cv::Mat_<cv::Vec3b> rescale_image_and_camera(cv::Mat_<cv::Vec3b>, cv::Mat&, Camera&);
