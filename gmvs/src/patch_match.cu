#include "cuda_utils.cuh"
#include "patch_match.cuh"

#define BLOCK_H 16
#define BLOCK_W 16
#define CUDA_THREADS 1024
#define MIN_BLOCKS_PER_SM 4
#define RANDOM_SEED 3407

__device__ float bilinear_interpolate(
    const float* __restrict__ input_ptr,
    const float2 pt,
    const int32_t height,
    const int32_t width) {
    // get the values of the four vertices
    const int32_t hl = clamp(static_cast<int32_t>(pt.y), 0, height - 1);
    const int32_t wl = clamp(static_cast<int32_t>(pt.x), 0, width - 1);
    const float h_diff = pt.y - static_cast<float>(hl);
    const float w_diff = pt.x - static_cast<float>(wl);

    const int32_t hu = (hl == height - 1) ? hl : hl + 1;
    const int32_t wu = (wl == width - 1) ? wl : wl + 1;

    const float v0 = input_ptr[hl * width + wl];
    const float v1 = input_ptr[hl * width + wu];
    const float v2 = input_ptr[hu * width + wl];
    const float v3 = input_ptr[hu * width + wu];

    // bilinear interpolation
    const float ret = lerp(lerp(v0, v1, w_diff), lerp(v2, v3, w_diff), h_diff);

    return ret;
}

__device__ void sort_small(float* __restrict__ d, const int32_t n) {
    int32_t j;
    for (int32_t i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) d[j] = d[j - 1];
        d[j] = tmp;
    }
}

__device__ int32_t find_min_cost_index(const float* __restrict__ costs, const int32_t n) {
    float min_cost = costs[0];
    int32_t min_cost_idx = 0;
    for (int32_t idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ void set_bit(uint32_t& input, const uint32_t n) { input |= (uint32_t)(1 << n); }

__device__ int32_t is_set(uint32_t input, const uint32_t n) { return (input >> n) & 1; }

__device__ void mat33_dot_vec3(const float mat[9], const float4 vec, float4* result) {
    result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
    result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float vec3_dot_vec3(const float4 vec1, const float4 vec2) {
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void normalize_vec3(float4* __restrict__ vec) {
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf(normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

__device__ void transform_PDF_to_CDF(float* __restrict__ probs, const int32_t num_probs) {
    float prob_sum = 0.0f;
    for (int32_t i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int32_t i = 0; i < num_probs; ++i) {
        const float prob = probs[i] * inv_prob_sum;
        cum_prob += prob;
        probs[i] = cum_prob;
    }
}

// get the point position in the camera coordinate
__device__ void get_3D_point(
    const Camera camera, const int2 p, const float depth, float* __restrict__ X) {
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}

__device__ float4 get_view_direction(const Camera camera, const int2 p, const float depth) {
    // get the 3d position of the current pixel according to the depth and
    // intrinsic
    float X[3];
    get_3D_point(camera, p, depth, X);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    // normalize the position as view direction
    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z = X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

__device__ float get_distance_to_origin(
    const Camera camera, const int2 p, const float depth, const float4 normal) {
    // get the 3d position of the current pixel according to the depth and
    // intrinsic
    float X[3];
    get_3D_point(camera, p, depth, X);
    // dot product the normal and point to get the distance to the coordinate
    // origin
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__ float compute_depth_from_plane_hypothesis(
    const Camera camera, const float4 plane_hypothesis, const int2 p) {
    return -plane_hypothesis.w * camera.K[0] /
           ((p.x - camera.K[2]) * plane_hypothesis.x +
            (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y +
            camera.K[0] * plane_hypothesis.z);
}

__device__ float4 generate_random_normal(
    const Camera camera, const int2 p, curandState* __restrict__ rand_state, const float depth) {
    // randomly generate the normal and store in the plane_hypothesis
    float4 plane_hypothesis;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    plane_hypothesis.x = 2.0f * q1 * sq;
    plane_hypothesis.y = 2.0f * q2 * sq;
    plane_hypothesis.z = 1.0f - 2.0f * s;
    plane_hypothesis.w = 0;

    // get the view direction in the camera coordinate
    float4 view_direction = get_view_direction(camera, p, depth);
    // assert the angle between the normal and the view direction is greater
    // than 90
    float dot_product = plane_hypothesis.x * view_direction.x +
                        plane_hypothesis.y * view_direction.y +
                        plane_hypothesis.z * view_direction.z;
    if (dot_product > 0.0f) {
        plane_hypothesis.x = -plane_hypothesis.x;
        plane_hypothesis.y = -plane_hypothesis.y;
        plane_hypothesis.z = -plane_hypothesis.z;
    }
    normalize_vec3(&plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 generate_perturbed_normal(
    const Camera camera,
    const int2 p,
    const float4 normal,
    curandState* __restrict__ rand_state,
    const float perturbation) {
    float4 view_direction = get_view_direction(camera, p, 1.0f);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    mat33_dot_vec3(R, normal, &normal_perturbed);

    if (vec3_dot_vec3(normal_perturbed, view_direction) >= 0.0f) {
        normal_perturbed = normal;
    }

    normalize_vec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ void compute_homography(
    const Camera ref_camera,
    const Camera src_camera,
    const float4 plane_hypothesis,
    float* __restrict__ H) {
    float ref_C[3];
    float src_C[3];
    ref_C[0] =
        -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] +
          ref_camera.R[6] * ref_camera.t[2]);
    ref_C[1] =
        -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] +
          ref_camera.R[7] * ref_camera.t[2]);
    ref_C[2] =
        -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] +
          ref_camera.R[8] * ref_camera.t[2]);
    src_C[0] =
        -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] +
          src_camera.R[6] * src_camera.t[2]);
    src_C[1] =
        -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] +
          src_camera.R[7] * src_camera.t[2]);
    src_C[2] =
        -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] +
          src_camera.R[8] * src_camera.t[2]);

    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] +
                    src_camera.R[2] * ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] +
                    src_camera.R[2] * ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] +
                    src_camera.R[2] * ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] +
                    src_camera.R[5] * ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] +
                    src_camera.R[5] * ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] +
                    src_camera.R[5] * ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] +
                    src_camera.R[8] * ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] +
                    src_camera.R[8] * ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] +
                    src_camera.R[8] * ref_camera.R[8];
    C_relative[0] = (ref_C[0] - src_C[0]);
    C_relative[1] = (ref_C[1] - src_C[1]);
    C_relative[2] = (ref_C[2] - src_C[2]);
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] +
                    src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] +
                    src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] +
                    src_camera.R[8] * C_relative[2];

    H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
    H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
    H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
    H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
    H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
    H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
    H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
    H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
    H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[4];
    tmp[2] =
        -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[4];
    tmp[5] =
        -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[4];
    tmp[8] =
        -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
    H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
    H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
    H[6] = src_camera.K[8] * tmp[6];
    H[7] = src_camera.K[8] * tmp[7];
    H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 compute_corresponding_point(const float* H, const int2 p) {
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 transform_normal(const Camera camera, float4 plane_hypothesis) {
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y +
                           camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y +
                           camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y +
                           camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 transform_normal_to_ref_cam(const Camera camera, float4 plane_hypothesis) {
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y +
                           camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y +
                           camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y +
                           camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 compute_plane_hypothesis_from_normal_depth(
    const float* ref_depth_ptr, const float* ref_normal_ptr, const Camera camera, const int2 p) {
    float4 plane_hypothesis =
        make_float4(ref_normal_ptr[0], ref_normal_ptr[1], ref_normal_ptr[2], ref_depth_ptr[0]);
    // rotate the normal(from the world coordinate to the camera coordinate)
    plane_hypothesis = transform_normal_to_ref_cam(camera, plane_hypothesis);
    // get the distance to the coordinate origin from depth
    float depth = plane_hypothesis.w;
    plane_hypothesis.w = get_distance_to_origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float compute_bilateral_weight(
    const float x_dist,
    const float y_dist,
    const float pix,
    const float center_pix,
    const float sigma_spatial,
    const float sigma_color) {
    // spatial_dist means the distance on the patch to the patch center
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(
        -spatial_dist / (2.0f * sigma_spatial * sigma_spatial) -
        color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float compute_bilateral_NCC(
    const float* __restrict__ ref_image,
    const Camera ref_camera,
    const float* __restrict__ src_image,
    const Camera src_camera,
    const int2 p,
    const float4 plane_hypothesis,
    const PatchMatchParams params) {
    const float cost_max = 2.0f;
    int32_t radius = params.patch_size / 2;

    // compute the homography matrix between two images via the camera
    // parameters
    float H[9];
    compute_homography(ref_camera, src_camera, plane_hypothesis, H);
    // get the corresponding pixel index on the src_image via homography matrix
    float2 pt = compute_corresponding_point(H, p);
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
        return cost_max;
    }

    const int32_t height = ref_camera.height;
    const int32_t width = ref_camera.width;
    float cost = 0.0f;
    {
        // init the immediate parameters
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float ref_center_pix = ref_image[p.y * width + p.x];

        // patch looping
        for (int32_t i = -radius; i < radius + 1; i += params.radius_increment) {
            for (int32_t j = -radius; j < radius + 1; j += params.radius_increment) {
                // get the index and gray value on the ref_image
                const int2 ref_pt =
                    make_int2(clamp(p.x + i, 0, width - 1), clamp(p.y + j, 0, height - 1));
                const float ref_pix = ref_image[ref_pt.y * width + ref_pt.x];
                // get the index and gray value on the src_image
                float2 src_pt = compute_corresponding_point(H, ref_pt);
                const float src_pix = bilinear_interpolate(src_image, src_pt, height, width);

                /* get the bilateral weight, which describes the photometric
                   consistency between the reference and source patch
                 */
                float weight = compute_bilateral_weight(
                    i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

                sum_ref += weight * ref_pix;
                sum_ref_ref += weight * ref_pix * ref_pix;
                sum_src += weight * src_pix;
                sum_src_src += weight * src_pix * src_pix;
                sum_ref_src += weight * ref_pix * src_pix;
                bilateral_weight_sum += weight;
            }
        }
        // average the values of the patch
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        // calculate the variance of the patch in ref_image and src_image
        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar) {  // for smooth area
            cost = cost_max;
        } else {
            // calculate the co-variance between ref_image and src_image
            // TODO: analyze here
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
        return cost;
    }
}

__device__ float compute_multi_view_initial_cost_and_selected_views(
    const float* __restrict__ ref_src_images,
    const Camera* __restrict__ ref_src_cameras,
    const int2 p,
    const float4 plane_hypothesis,
    uint32_t* __restrict__ selected_views,
    const PatchMatchParams params) {
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int32_t cost_count = 0;
    int32_t num_valid_views = 0;

    const int32_t height = ref_src_cameras[0].height;
    const int32_t width = ref_src_cameras[0].width;

    for (int32_t i = 1; i < params.num_images; ++i) {
        // NCC cost
        float c = compute_bilateral_NCC(
            ref_src_images,
            ref_src_cameras[0],
            ref_src_images + i * height * width,
            ref_src_cameras[i],
            p,
            plane_hypothesis,
            params);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    // find the top_k match view and calculate the average cost
    int32_t top_k = min(num_valid_views, params.top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int32_t i = 0; i < top_k; ++i) {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int32_t i = 0; i < params.num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                set_bit(*selected_views, i);
            }
        }
        return cost / top_k;
    } else {
        return cost_max;
    }
}

__device__ void compute_multi_view_cost_vector(
    const float* __restrict__ ref_src_images,
    const Camera* __restrict__ ref_src_cameras,
    const int2 p,
    const float4 plane_hypothesis,
    float* __restrict__ cost_vector,
    const PatchMatchParams params) {
    const int32_t height = ref_src_cameras[0].height;
    const int32_t width = ref_src_cameras[0].width;
    for (int32_t i = 1; i < params.num_images; ++i) {
        cost_vector[i - 1] = compute_bilateral_NCC(
            ref_src_images,
            ref_src_cameras[0],
            ref_src_images + i * height * width,
            ref_src_cameras[i],
            p,
            plane_hypothesis,
            params);
    }
}

__device__ float3
get_3D_point_on_World(const float x, const float y, const float depth, const Camera camera) {
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

__device__ void project_on_camera(
    const float3 PointX, const Camera camera, float2& point, float& depth) {
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

__device__ float compute_geom_consistency_cost(
    const float* __restrict__ src_depths,
    const Camera ref_camera,
    const Camera src_camera,
    const float4 plane_hypothesis,
    const int2 p) {
    const float max_cost = 5.0f;

    float depth = compute_depth_from_plane_hypothesis(ref_camera, plane_hypothesis, p);
    float3 forward_point = get_3D_point_on_World(p.x, p.y, depth, ref_camera);

    float2 src_pt;
    float src_d;
    project_on_camera(forward_point, src_camera, src_pt, src_d);
    const int32_t height = ref_camera.height;
    const int32_t width = ref_camera.width;
    const float src_depth = bilinear_interpolate(src_depths, src_pt, height, width);

    if (src_depth == 0.0f) {
        return max_cost;
    }

    float3 src_3D_pt = get_3D_point_on_World(src_pt.x, src_pt.y, src_depth, src_camera);

    float2 backward_point;
    float ref_d;
    project_on_camera(src_3D_pt, ref_camera, backward_point, ref_d);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__device__ void plane_hypothesis_refinement(
    const float* __restrict__ ref_src_images,
    float* __restrict__ ref_src_depths,
    const Camera* __restrict__ ref_src_cameras,
    float4* __restrict__ plane_hypothesis,
    float* __restrict__ depth,
    float* __restrict__ cost,
    curandState* __restrict__ rand_state,
    const float* __restrict__ view_weights,
    const float weight_norm,
    const int2 p,
    const PatchMatchParams params) {
    float perturbation = 0.02f;

    float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
    float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
    float angle_sigma = M_PI * (5.0f / 180.0f);
    float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;

    float depth_rand;
    float4 plane_hypothesis_rand;

    const Camera ref_camera = ref_src_cameras[0];
    const int32_t height = ref_camera.height;
    const int32_t width = ref_camera.width;

    depth_rand =
        curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
    plane_hypothesis_rand = generate_random_normal(ref_camera, p, rand_state, *depth);
    float depth_perturbed = *depth;
    const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
    const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
    do {
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) +
                          depth_min_perturbed;
    } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);
    float4 plane_hypothesis_perturbed = generate_perturbed_normal(
        ref_camera, p, *plane_hypothesis, rand_state, perturbation * M_PI);

    const int32_t num_planes = 5;
    float depth_candidates[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
    float4 normal_candidates[num_planes] = {*plane_hypothesis,
                                            plane_hypothesis_rand,
                                            plane_hypothesis_rand,
                                            plane_hypothesis_perturbed,
                                            *plane_hypothesis};

    for (int32_t i = 0; i < num_planes; ++i) {
        float cost_vector[32] = {2.0f};
        float4 temp_plane_hypothesis = normal_candidates[i];
        temp_plane_hypothesis.w = get_distance_to_origin(
            ref_camera,
            p,
            depth_candidates[i],
            temp_plane_hypothesis);  // dists[i];
        compute_multi_view_cost_vector(
            ref_src_images, ref_src_cameras, p, temp_plane_hypothesis, cost_vector, params);

        float temp_cost = 0.0f;
        for (int32_t j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                temp_cost += view_weights[j] *
                             (cost_vector[j] + 0.1f * compute_geom_consistency_cost(
                                                          ref_src_depths + (j + 1) * height * width,
                                                          ref_camera,
                                                          ref_src_cameras[j + 1],
                                                          temp_plane_hypothesis,
                                                          p));
            }
        }
        temp_cost /= weight_norm;

        float depth_before =
            compute_depth_from_plane_hypothesis(ref_camera, temp_plane_hypothesis, p);
        if (depth_before >= params.depth_min && depth_before <= params.depth_max &&
            temp_cost < *cost) {
            *depth = depth_before;
            *plane_hypothesis = temp_plane_hypothesis;
            *cost = temp_cost;
        }
    }
}

__device__ void checkerboard_propagation(
    const float* __restrict__ ref_src_images,
    float* __restrict__ ref_src_depths,
    const Camera* __restrict__ cameras,
    float4* __restrict__ plane_hypotheses,
    float* __restrict__ costs,
    curandState* __restrict__ rand_states,
    uint32_t* __restrict__ selected_views,
    float* __restrict__ view_weights,
    float* __restrict__ weight_norm,
    float* __restrict__ depth_now,
    const int2 p,
    const PatchMatchParams params,
    const int32_t iter) {
    /* get the pixel and query indices
                o u_f
                |
                o u_n
      l_f  l_n  |  r_n  r_f
        o---o---c---o---o
                |
                o d_n
                |
                o d_f
     */
    const Camera ref_camera = cameras[0];
    const int32_t height = ref_camera.height;
    const int32_t width = ref_camera.width;
    const int32_t center = p.y * width + p.x;
    int32_t left_near = center - 1;
    int32_t left_far = center - 3;
    int32_t right_near = center + 1;
    int32_t right_far = center + 3;
    int32_t up_near = center - width;
    int32_t up_far = center - 3 * width;
    int32_t down_near = center + width;
    int32_t down_far = center + 3 * width;

    // Adaptive Checkerboard Sampling
    float cost_array[8][32] = {2.0f};
    // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near,
    // 5 -- left_far, 6 -- right_near, 7 -- right_far
    bool flag[8] = {false};

    float cost_min;
    int32_t cost_min_point;

    // up_far
    if (p.y > 2) {  // out of bound judgement
        flag[1] = true;
        cost_min = costs[up_far];
        cost_min_point = up_far;
        // query the minimum cost along the vertical direction
        for (int32_t i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int32_t point_temp = up_far - 2 * i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        up_far = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[up_far], cost_array[1], params);
    }

    // dwon_far
    if (p.y < height - 3) {  // out of bound judgement
        flag[3] = true;
        cost_min = costs[down_far];
        cost_min_point = down_far;
        // query the minimum cost along the vertical direction
        for (int32_t i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int32_t point_temp = down_far + 2 * i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        down_far = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[down_far], cost_array[3], params);
    }

    // left_far
    if (p.x > 2) {  // out of bound judgement
        flag[5] = true;
        cost_min = costs[left_far];
        cost_min_point = left_far;
        // query the minimum cost along the horizontal direction
        for (int32_t i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int32_t point_temp = left_far - 2 * i;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        left_far = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[left_far], cost_array[5], params);
    }

    // right_far
    if (p.x < width - 3) {  // out of bound judgement
        flag[7] = true;
        cost_min = costs[right_far];
        cost_min_point = right_far;
        // query the minimum cost along the horizontal direction
        for (int32_t i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int32_t point_temp = right_far + 2 * i;
                if (cost_min < costs[point_temp]) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        right_far = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[right_far], cost_array[7], params);
    }

    // up_near
    if (p.y > 0) {  // out of bound judgement
        flag[0] = true;
        cost_min = costs[up_near];
        cost_min_point = up_near;
        // query the minimum cost along the v-type direction
        for (int32_t i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int32_t point_temp = up_near - (1 + i) * width - i;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
            if (p.y > 1 + i && p.x < width - 1 - i) {
                int32_t point_temp = up_near - (1 + i) * width + i;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        up_near = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[up_near], cost_array[0], params);
    }

    // down_near
    if (p.y < height - 1) {  // out of bound judgement
        flag[2] = true;
        cost_min = costs[down_near];
        cost_min_point = down_near;
        // query the minimum cost along the v-type direction
        for (int32_t i = 0; i < 3; ++i) {
            if (p.y < height - 2 - i && p.x > i) {
                int32_t point_temp = down_near + (1 + i) * width - i;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
            if (p.y < height - 2 - i && p.x < width - 1 - i) {
                int32_t point_temp = down_near + (1 + i) * width + i;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        down_near = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[down_near], cost_array[2], params);
    }

    // left_near
    if (p.x > 0) {  // out of bound judgement
        flag[4] = true;
        cost_min = costs[left_near];
        cost_min_point = left_near;
        // query the minimum cost along the v-type direction
        for (int32_t i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int32_t point_temp = left_near - (1 + i) - i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
            if (p.x > 1 + i && p.y < height - 1 - i) {
                int32_t point_temp = left_near - (1 + i) + i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        left_near = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[left_near], cost_array[4], params);
    }

    // right_near
    if (p.x < width - 1) {  // out of bound judgement
        flag[6] = true;
        cost_min = costs[right_near];
        cost_min_point = right_near;
        // query the minimum cost along the v-type direction
        for (int32_t i = 0; i < 3; ++i) {
            if (p.x < width - 2 - i && p.y > i) {
                int32_t point_temp = right_near + (1 + i) - i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
            if (p.x < width - 2 - i && p.y < height - 1 - i) {
                int32_t point_temp = right_near + (1 + i) + i * width;
                if (costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
        right_near = cost_min_point;
        compute_multi_view_cost_vector(
            ref_src_images, cameras, p, plane_hypotheses[right_near], cost_array[6], params);
    }
    const int32_t positions[8] = {
        up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Multi-hypothesis Joint View Selection
    // float view_weights[32] = {0.0f};
    float view_selection_priors[32] = {0.0f};
    int32_t neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
    for (int32_t i = 0; i < 4; ++i) {
        if (flag[2 * i]) {  // query the 4-nearest pixel
            // accumulate the view selection score
            for (int32_t j = 0; j < params.num_images - 1; ++j) {
                if (is_set(selected_views[neighbor_positions[i]], j) == 1) {
                    view_selection_priors[j] += 0.9f;
                } else {
                    view_selection_priors[j] += 0.1f;
                }
            }
        }
    }

    // calculate the PDF
    float sampling_probs[32] = {0.0f};
    float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
    for (int32_t i = 0; i < params.num_images - 1; i++) {
        float count = 0;
        int32_t count_false = 0;
        float tmpw = 0;
        // statistic the cost_array for checkerboard
        for (int32_t j = 0; j < 8; j++) {
            if (cost_array[j][i] < cost_threshold) {
                tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                count++;
            }
            if (cost_array[j][i] > 1.2f) {
                count_false++;
            }
        }
        if (count > 2 && count_false < 3) {
            sampling_probs[i] = tmpw / count;
        } else if (count_false < 3) {
            sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
        }
        sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
    }

    // PDF to CDF
    transform_PDF_to_CDF(sampling_probs, params.num_images - 1);
    for (int32_t sample = 0; sample < 15; ++sample) {
        const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

        for (int32_t image_id = 0; image_id < params.num_images - 1; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    // store the selected views and their weights
    uint32_t temp_selected_views = 0;
    int32_t num_selected_view = 0;
    *weight_norm = 0;
    for (int32_t i = 0; i < params.num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            set_bit(temp_selected_views, i);
            *weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    // calculate the final cost for 8 query pixels
    float final_costs[8] = {0.0f};
    for (int32_t i = 0; i < 8; ++i) {
        for (int32_t j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (flag[i]) {
                    final_costs[i] +=
                        view_weights[j] *
                        (cost_array[i][j] + 0.1f * compute_geom_consistency_cost(
                                                       ref_src_depths + (j + 1) * height * width,
                                                       ref_camera,
                                                       cameras[j + 1],
                                                       plane_hypotheses[positions[i]],
                                                       p));
                } else {
                    final_costs[i] += view_weights[j] * cost_array[i][j];
                }
            }
        }
        final_costs[i] /= *weight_norm;
    }

    // find the index of the cost minimum
    const int32_t min_cost_idx = find_min_cost_index(final_costs, 8);

    // find the cost of the current pixel
    float cost_vector_now[32] = {2.0f};
    compute_multi_view_cost_vector(
        ref_src_images, cameras, p, plane_hypotheses[center], cost_vector_now, params);
    float cost_now = 0.0f;
    for (int32_t i = 0; i < params.num_images - 1; ++i) {
        cost_now += view_weights[i] *
                    (cost_vector_now[i] + 0.1f * compute_geom_consistency_cost(
                                                     ref_src_depths + (i + 1) * height * width,
                                                     ref_camera,
                                                     cameras[i + 1],
                                                     plane_hypotheses[center],
                                                     p));
    }
    // get the average of cost as the cost of the current pixel
    cost_now /= *weight_norm;
    costs[center] = cost_now;
    // get the depth according to the plane hypothesis
    *depth_now = compute_depth_from_plane_hypothesis(ref_camera, plane_hypotheses[center], p);

    if (flag[min_cost_idx]) {
        const int32_t pos = positions[min_cost_idx];
        float depth_before =
            compute_depth_from_plane_hypothesis(ref_camera, plane_hypotheses[pos], p);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max &&
            final_costs[min_cost_idx] < cost_now) {
            *depth_now = depth_before;
            plane_hypotheses[center] = plane_hypotheses[pos];
            costs[center] = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }
}

// kernels

// __launch_bounds__(CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void initialization(
    const float* __restrict__ ref_src_images,
    const float* __restrict__ ref_src_depths,
    const float* __restrict__ ref_src_normals,
    Camera* __restrict__ cameras_cuda,      // num_images
    float4* __restrict__ plane_hypotheses,  // [H, W]
    float* __restrict__ costs,              // [H, W]
    curandState* __restrict__ rand_states,  // [H, W]
    uint32_t* __restrict__ selected_views,  // [H, W]
    const PatchMatchParams params) {
    // the pixel index
    const int2 p =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int32_t width = cameras_cuda[0].width;
    int32_t height = cameras_cuda[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    // 1d index
    const int32_t center = p.y * width + p.x;
    // curand_init(clock64(), p.y, p.x, &rand_states[center]);
    curand_init(RANDOM_SEED, p.y, p.x, &rand_states[center]);

    // get the plane hypothesis for the current pixel
    const int32_t img_offset = p.y * width + p.x;

    float4 plane_hypothesis = compute_plane_hypothesis_from_normal_depth(
        ref_src_depths + img_offset, ref_src_normals + img_offset * 3, cameras_cuda[0], p);

    // rewrite the plane hypothesis for the current pixel
    plane_hypotheses[center] = plane_hypothesis;
    // calculate the cost of the current pixel via NCC
    costs[center] = compute_multi_view_initial_cost_and_selected_views(
        ref_src_images, cameras_cuda, p, plane_hypotheses[center], &selected_views[center], params);
}

// __launch_bounds__(CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void black_red_pixel_update(
    const float* __restrict__ ref_src_images,
    float* __restrict__ ref_src_depths,
    Camera* __restrict__ ref_src_cameras,   // num_images
    float4* __restrict__ plane_hypotheses,  // [H, W]
    float* __restrict__ costs,              // [H, W]
    curandState* __restrict__ rand_states,  // [H, W]
    uint32_t* __restrict__ selected_views,  // [H, W]
    const PatchMatchParams params,
    const int32_t iter,
    const bool black) {
    // get the pixel index for black
    int2 p =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = black ? p.y * 2 : p.y * 2 + 1;
    } else {
        p.y = black ? p.y * 2 + 1 : p.y * 2;
    }
    int32_t width = ref_src_cameras[0].width;
    int32_t height = ref_src_cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    // perform checkerboard propagation
    float view_weights[32] = {0.0f};
    float weight_norm;
    float depth_now;
    checkerboard_propagation(
        ref_src_images,
        ref_src_depths,
        ref_src_cameras,
        plane_hypotheses,  // output
        costs,             // output
        rand_states,
        selected_views,  // output
        view_weights,    // output
        &weight_norm,    // output
        &depth_now,      // output
        p,
        params,
        iter);

    const int32_t center = p.y * width + p.x;
    plane_hypothesis_refinement(
        ref_src_images,
        ref_src_depths,
        ref_src_cameras,
        &plane_hypotheses[center],  // output
        &depth_now,                 // output
        &costs[center],             // output
        &rand_states[center],
        view_weights,
        weight_norm,
        p,
        params);
}

// __launch_bounds__(CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void get_depth_and_normal(
    Camera* __restrict__ cameras,
    float4* __restrict__ plane_hypotheses,
    const PatchMatchParams params) {
    const int2 p =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int32_t width = cameras[0].width;
    const int32_t height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int32_t center = p.y * width + p.x;
    plane_hypotheses[center].w =
        compute_depth_from_plane_hypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = transform_normal(cameras[0], plane_hypotheses[center]);
}

// __launch_bounds__(CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void black_red_pixel_filter(
    const Camera* __restrict__ ref_src_cameras,
    float4* __restrict__ plane_hypotheses,
    float* __restrict__ costs,
    const bool black) {
    int2 p =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = black ? p.y * 2 : p.y * 2 + 1;
    } else {
        p.y = black ? p.y * 2 + 1 : p.y * 2;
    }

    const Camera ref_camera = ref_src_cameras[0];
    const int32_t width = ref_camera.width;
    const int32_t height = ref_camera.height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int32_t center = p.y * width + p.x;

    float filter[21];
    int32_t index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int32_t left = center - 1;
    const int32_t leftleft = center - 3;

    // Up
    const int32_t up = center - width;
    const int32_t upup = center - 3 * width;

    // Down
    const int32_t down = center + width;
    const int32_t downdown = center + 3 * width;

    // Right
    const int32_t right = center + 1;
    const int32_t rightright = center + 3;

    if (costs[center] < 0.001f) {
        return;
    }

    if (p.y > 0) {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y > 2) {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y > 4) {
        filter[index++] = plane_hypotheses[upup - width * 2].w;
    }
    if (p.y < height - 1) {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y < height - 3) {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y < height - 5) {
        filter[index++] = plane_hypotheses[downdown + width * 2].w;
    }
    if (p.x > 0) {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x > 2) {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x > 4) {
        filter[index++] = plane_hypotheses[leftleft - 2].w;
    }
    if (p.x < width - 1) {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x < width - 3) {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x < width - 5) {
        filter[index++] = plane_hypotheses[rightright + 2].w;
    }
    if (p.y > 0 && p.x < width - 2) {
        filter[index++] = plane_hypotheses[up + 2].w;
    }
    if (p.y < height - 1 && p.x < width - 2) {
        filter[index++] = plane_hypotheses[down + 2].w;
    }
    if (p.y > 0 && p.x > 1) {
        filter[index++] = plane_hypotheses[up - 2].w;
    }
    if (p.y < height - 1 && p.x > 1) {
        filter[index++] = plane_hypotheses[down - 2].w;
    }
    if (p.x > 0 && p.y > 2) {
        filter[index++] = plane_hypotheses[left - width * 2].w;
    }
    if (p.x < width - 1 && p.y > 2) {
        filter[index++] = plane_hypotheses[right - width * 2].w;
    }
    if (p.x > 0 && p.y < height - 2) {
        filter[index++] = plane_hypotheses[left + width * 2].w;
    }
    if (p.x < width - 1 && p.y < height - 2) {
        filter[index++] = plane_hypotheses[right + width * 2].w;
    }

    // average the depth
    sort_small(filter, index);
    int32_t median_index = index / 2;
    if (index % 2 == 0) {
        plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
    } else {
        plane_hypotheses[center].w = filter[median_index];
    }
}

// __launch_bounds__(CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void export_plane_hypotheses_cost(
    const float4* __restrict__ plane_ptr,  // [H, W]
    const float* __restrict__ cost_ptr,    // [H, W]
    const int32_t height,
    const int32_t width,
    // output
    float* __restrict__ output_depth_ptr,   // [H, W]
    float* __restrict__ output_normal_ptr,  // [H, W, 3]
    float* __restrict__ output_cost_ptr     // [H, W]
) {
    // const int32_t row = blockIdx.x;
    // const int32_t col = threadIdx.x;

    // const int32_t index = row * width + col;

    const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= height * width){
        return;
    } 
    const int32_t row = pixel_id / width;
    const int32_t col = pixel_id % width;
    
    const int32_t index = row * width + col;

    output_depth_ptr[index] = plane_ptr[index].w;
    output_normal_ptr[index * 3 + 0] = plane_ptr[index].x;
    output_normal_ptr[index * 3 + 1] = plane_ptr[index].y;
    output_normal_ptr[index * 3 + 2] = plane_ptr[index].z;
    output_cost_ptr[index] = cost_ptr[index];
}

mvs::PatchMatcher::PatchMatcher() {
    num_images = 0;
    params = PatchMatchParams();
}
mvs::PatchMatcher::~PatchMatcher() {
}

void mvs::PatchMatcher::add_samples(
    const vector<Problem> problems,
    const vector<Camera> cameras,
    const Tensor& images,
    const Tensor& depths,
    const Tensor& normals,
    const Tensor& costs) {
    // check torch tensor
    CHECK_CPU_INPUT(images);
    CHECK_CPU_INPUT(depths);
    CHECK_CPU_INPUT(normals);
    CHECK_CPU_INPUT(costs);

    // get the number of images
    num_images = problems.size();
    cameras_host = cameras;
    problems_host = problems;

    height = cameras[0].height;
    width = cameras[0].width;

    // allocate to copy cuda
    images_host = images;
    depths_host = depths;
    normals_host = normals;
    costs_host = costs;
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

// launch function

tuple<Tensor, Tensor, Tensor> mvs::PatchMatcher::run_patch_match(
    const int32_t problem_idx, const bool verbose) {
    // define the grid and block for cuda parallelization
    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y = (height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    // half pixels
    dim3 grid_size_checkerboard;
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard;
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    // set parameters
    const Problem problem_host = problems_host[problem_idx];
    params.num_images = problem_host.num_ngb + 1;
    const int32_t ref_id = problem_host.ref_image_id;
    params.depth_min = cameras_host[ref_id].depth_min * 0.6;
    params.depth_max = cameras_host[ref_id].depth_max * 1.2;

    // cuda init
    float4* plane_hypotheses_cuda = NULL;
    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (height * width));
    curandState* rand_states_cuda = NULL;
    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (height * width));
    uint32_t* selected_views_cuda = NULL;
    cudaMalloc((void**)&selected_views_cuda, sizeof(uint32_t) * (height * width));

    Tensor cost_cuda = costs_host.index({ref_id}).to(torch::kCUDA);
    Tensor images_cuda = torch::zeros(
        {params.num_images, height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor depths_cuda = torch::zeros(
        {params.num_images, height, width},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor normals_cuda = torch::zeros(
        {params.num_images, height, width, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Camera* cameras_cuda = NULL;
    {
        images_cuda.index_put_({0}, images_host.index({ref_id}).to(torch::kCUDA));
        depths_cuda.index_put_({0}, depths_host.index({ref_id}).to(torch::kCUDA));
        normals_cuda.index_put_({0}, normals_host.index({ref_id}).to(torch::kCUDA));

        const int32_t* src_ids = problem_host.src_image_ids;
        vector<Camera> ref_src_cameras;
        ref_src_cameras.push_back(cameras_host[ref_id]);
        for (uint16_t i = 0; i < problem_host.num_ngb; ++i) {
            const int32_t src_id = src_ids[i];
            images_cuda.index_put_({i + 1}, images_host.index({src_id}).to(torch::kCUDA));
            depths_cuda.index_put_({i + 1}, depths_host.index({src_id}).to(torch::kCUDA));
            normals_cuda.index_put_({i + 1}, normals_host.index({src_id}).to(torch::kCUDA));
            ref_src_cameras.push_back(cameras_host[src_id]);
        }
        cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * params.num_images);
        cudaMemcpy(
            cameras_cuda,
            &ref_src_cameras[0],
            sizeof(Camera) * params.num_images,
            cudaMemcpyHostToDevice);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    initialization<<<grid_size_randinit, block_size_randinit>>>(
        images_cuda.data_ptr<float>(),
        depths_cuda.data_ptr<float>(),
        normals_cuda.data_ptr<float>(),
        cameras_cuda,                 // num_images
        plane_hypotheses_cuda,        // [H, W]
        cost_cuda.data_ptr<float>(),  // [H, W]
        rand_states_cuda,             // [H, W]
        selected_views_cuda,          // [H, W]
        params);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    for (int32_t i = 0; i < params.max_iterations; ++i) {
        // black pixel update
        black_red_pixel_update<<<grid_size_checkerboard, block_size_checkerboard>>>(
            images_cuda.data_ptr<float>(),
            depths_cuda.data_ptr<float>(),
            cameras_cuda,                 // num_images
            plane_hypotheses_cuda,        // [H, W]
            cost_cuda.data_ptr<float>(),  // [H, W]
            rand_states_cuda,             // [H, W]
            selected_views_cuda,          // [H, W]
            params,
            i,
            true);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        // red pixel update
        black_red_pixel_update<<<grid_size_checkerboard, block_size_checkerboard>>>(
            images_cuda.data_ptr<float>(),
            depths_cuda.data_ptr<float>(),
            cameras_cuda,                 // num_images
            plane_hypotheses_cuda,        // [H, W]
            cost_cuda.data_ptr<float>(),  // [H, W]
            rand_states_cuda,             // [H, W]
            selected_views_cuda,          // [H, W]
            params,
            i,
            false);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        if (verbose) {
            std::cout << "iteration: " << i << std::endl;
        }
    }

    // get the depth and normal in the world coordinate
    get_depth_and_normal<<<grid_size_randinit, block_size_randinit>>>(
        cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    // average(smooth) the depth according to the black-red checkerboard
    // black pixel filter
    black_red_pixel_filter<<<grid_size_checkerboard, block_size_checkerboard>>>(
        cameras_cuda, plane_hypotheses_cuda, cost_cuda.data_ptr<float>(), true);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    // red pixel filter
    black_red_pixel_filter<<<grid_size_checkerboard, block_size_checkerboard>>>(
        cameras_cuda, plane_hypotheses_cuda, cost_cuda.data_ptr<float>(), false);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    Tensor output_depths = torch::zeros(
        {height, width}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor output_normals = torch::zeros(
        {height, width, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor output_costs = torch::zeros(
        {height, width}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    // export_plane_hypotheses_cost<<<height, width>>>(
    //     plane_hypotheses_cuda,
    //     cost_cuda.data_ptr<float>(),  // [H, W]
    //     height,
    //     width,
    //     // output
    //     output_depths.data_ptr<float>(),
    //     output_normals.data_ptr<float>(),
    //     output_costs.data_ptr<float>());
    const int32_t num_threads = 256;
    const int32_t num_blocks = (height * width - 1) / num_threads + 1;
    export_plane_hypotheses_cost<<<num_blocks, num_threads>>>(
        plane_hypotheses_cuda,
        cost_cuda.data_ptr<float>(),  // [H, W]
        height,
        width,
        // output
        output_depths.data_ptr<float>(),
        output_normals.data_ptr<float>(),
        output_costs.data_ptr<float>());

    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);

    return std::make_tuple(output_depths, output_normals, output_costs);
}