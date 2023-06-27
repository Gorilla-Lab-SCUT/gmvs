#include <torch/extension.h>

#include "cuda_utils.cuh"
#include "fusion.cuh"

__device__ float3 get_3D_point_on_world_cu(
    const float x,
    const float y,
    const float depth,
    const mvs::Camera camera) {
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

__device__ void project_on_camera_cu(
    const float3 PointX,
    const mvs::Camera camera,
    float2& point,
    float& depth) {
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

__device__ float get_angle_cu(const float3 v1, const float3 v2) {
    float dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    float angle = acosf(dot_product);
    // if angle is not a number the dot product was 1 and thus the two vectors
    // should be identical --> return 0
    if (angle != angle)
        return 0.0f;

    return angle;
}

__global__ void batch_fusion_kernel(
    const float* __restrict__ ref_src_depths,
    const float* __restrict__ ref_src_normals,
    bool* __restrict__ masks,
    const mvs::Camera* __restrict__ ref_src_cameras,
    const mvs::Problem* __restrict__ problems,
    const int32_t geom_consistent,
    const int32_t height,
    const int32_t width,
    // output
    float* __restrict__ output_proj_depth, // [H, W]
    float* __restrict__ output_proj_normal // [H, W, 3]
) {
    const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= height * width) {
        return;
    }
    const int32_t ref_r = pixel_id / width;
    const int32_t ref_c = pixel_id % width;
    const mvs::Problem problem = problems[0];

    // fullfill the output by 0
    const int32_t ref_index = ref_r * width + ref_c;
    int2 used_list[MAX_NGB];
    for (int32_t i = 0; i < problem.num_ngb; ++i) {
        used_list[i] = make_int2(-1, -1);
    }

    if (masks[problem.ref_image_id * height * width + ref_index])
        return;

    const float ref_depth = ref_src_depths[ref_index];
    const float3 ref_normal = make_float3(
        ref_src_normals[ref_index * 3 + 0],
        ref_src_normals[ref_index * 3 + 1],
        ref_src_normals[ref_index * 3 + 2]);

    if (ref_depth <= 0)
        return;

    const mvs::Camera ref_camera = ref_src_cameras[0];
    float3 ref_point = get_3D_point_on_world_cu(ref_c, ref_r, ref_depth, ref_camera);
    float3 consistent_point = ref_point;
    float3 consistent_normal = ref_normal;
    int32_t num_consistent = 0;

    for (int32_t j = 0; j < problem.num_ngb; ++j) {
        const int32_t src_offset = (j + 1) * height * width;
        mvs::Camera src_camera = ref_src_cameras[j + 1];
        float2 point;
        float proj_depth;
        project_on_camera_cu(ref_point, src_camera, point, proj_depth);
        const int32_t src_r = int32_t(point.y + 0.5f), src_c = int32_t(point.x + 0.5f);
        if (src_c >= 0 && src_c < width && src_r >= 0 && src_r < height) {
            const int32_t src_index = src_r * width + src_c;

            const float src_depth = ref_src_depths[src_offset + src_index];
            const float3 src_normal = make_float3(
                ref_src_normals[(src_offset + src_index) * 3 + 0],
                ref_src_normals[(src_offset + src_index) * 3 + 1],
                ref_src_normals[(src_offset + src_index) * 3 + 2]);
            if (src_depth <= 0.0) {
                continue;
            }

            float3 src_point = get_3D_point_on_world_cu(src_c, src_r, src_depth, src_camera);
            float2 tmp_pt;
            project_on_camera_cu(src_point, ref_camera, tmp_pt, proj_depth);
            float reproj_error = sqrt(pow(ref_c - tmp_pt.x, 2) + pow(ref_r - tmp_pt.y, 2));
            float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
            float angle = get_angle_cu(ref_normal, src_normal);

            if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                consistent_point.x += src_point.x;
                consistent_point.y += src_point.y;
                consistent_point.z += src_point.z;
                consistent_normal.x += src_normal.x;
                consistent_normal.y += src_normal.y;
                consistent_normal.z += src_normal.z;

                used_list[j].x = src_c;
                used_list[j].y = src_r;
                ++num_consistent;
            }
        }
    }
    __syncthreads();

    if (num_consistent >= geom_consistent) {
        consistent_point.x /= (num_consistent + 1.0f);
        consistent_point.y /= (num_consistent + 1.0f);
        consistent_point.z /= (num_consistent + 1.0f);
        consistent_normal.x /= (num_consistent + 1.0f);
        consistent_normal.y /= (num_consistent + 1.0f);
        consistent_normal.z /= (num_consistent + 1.0f);

        // get valid depth and normal
        float2 proj_point;
        float proj_depth;
        project_on_camera_cu(consistent_point, ref_camera, proj_point, proj_depth);
        const int32_t proj_ref_r = int32_t(proj_point.y + 0.5f),
                      proj_ref_c = int32_t(proj_point.x + 0.5f);

        if (proj_ref_c >= 0 && proj_ref_c < width && proj_ref_r >= 0 && proj_ref_r < height &&
            proj_depth > 0.001f) {
            const int32_t proj_index = proj_ref_r * width + proj_ref_c;
            output_proj_depth[proj_index] = proj_depth;
            output_proj_normal[proj_index * 3 + 0] = consistent_normal.x;
            output_proj_normal[proj_index * 3 + 1] = consistent_normal.y;
            output_proj_normal[proj_index * 3 + 2] = consistent_normal.z;
        }
        const int32_t* src_ids = problem.src_image_ids;
        for (int j = 0; j < problem.num_ngb; ++j) {
            if (used_list[j].x == -1)
                continue;
            const int32_t offset =
                src_ids[j] * height * width + used_list[j].y * width + used_list[j].x;
            masks[offset] = true;
        }
    }
    __syncthreads();
}



void mvs::Fuser::load_samples(
    const vector<Problem> problems,
    const vector<Camera> cameras,
    const Tensor& depths,
    const Tensor& normals,
    const Tensor& masks) {
    // check torch tensor
    CHECK_CPU_INPUT(depths);
    CHECK_CPU_INPUT(normals);
    CHECK_INPUT(masks);

    // get the number of images
    cameras_host = cameras;
    problems_host = problems;
    masks_cuda = masks;

    height = cameras[0].height;
    width = cameras[0].width;

    // allocate to copy cuda
    depths_host = depths;
    normals_host = normals;
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

tuple<Tensor, Tensor> mvs::Fuser::run_fusion(
    const int32_t problem_idx,
    const int32_t geom_consistent) {
    // malloc the cuda memory and store the camera/problem parameters as
    // gpu_cameras/gpu_problems
    mvs::Camera* gpu_cameras = NULL;
    mvs::Problem* gpu_problems = NULL;

    const int32_t num_images = problems_host[problem_idx].num_ngb + 1;
    Tensor depths_cuda = torch::zeros(
        {num_images, height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor normals_cuda = torch::zeros(
        {num_images, height, width, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    vector<Camera> ref_src_cameras;
    const Problem problem_host = problems_host[problem_idx];
    const int32_t ref_id = problem_host.ref_image_id;
    const int32_t* src_ids = problem_host.src_image_ids;
    ref_src_cameras.push_back(cameras_host[ref_id]);
    depths_cuda.index_put_({0}, depths_host.index({ref_id}).to(torch::kCUDA));
    normals_cuda.index_put_({0}, normals_host.index({ref_id}).to(torch::kCUDA));
    for (uint16_t i = 0; i < problem_host.num_ngb; ++i) {
        const int32_t src_id = src_ids[i];
        depths_cuda.index_put_({i + 1}, depths_host.index({src_id}).to(torch::kCUDA));
        normals_cuda.index_put_({i + 1}, normals_host.index({src_id}).to(torch::kCUDA));
        ref_src_cameras.push_back(cameras_host[src_id]);
    }

    cudaMalloc((void**)&gpu_cameras, sizeof(mvs::Camera) * num_images);
    cudaMemcpy(
        gpu_cameras, &ref_src_cameras[0], sizeof(mvs::Camera) * num_images, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_problems, sizeof(mvs::Problem));
    cudaMemcpy(
        gpu_problems, &problems_host[problem_idx], sizeof(mvs::Problem), cudaMemcpyHostToDevice);

    Tensor proj_depth = torch::zeros(
        {height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor proj_normal = torch::zeros(
        {height, width, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    const int32_t num_thread = 256;
    const int32_t num_blocks = (height * width - 1) / num_thread + 1;
    batch_fusion_kernel<<<num_blocks, num_thread>>>(
        depths_cuda.data_ptr<float>(),
        normals_cuda.data_ptr<float>(),
        masks_cuda.data_ptr<bool>(),
        gpu_cameras,
        gpu_problems,
        geom_consistent,
        height,
        width,
        proj_depth.data_ptr<float>(),
        proj_normal.data_ptr<float>());

    tuple<Tensor, Tensor> results = std::make_tuple(proj_depth, proj_normal);

    cudaFree(gpu_cameras);
    cudaFree(gpu_problems);

    return results;
}