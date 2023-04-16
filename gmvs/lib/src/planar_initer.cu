#include <algorithm>
#include "planar_initer.cuh"

inline bool has_ending(std::string const& full_string, std::string const& ending) {
    if (full_string.length() >= ending.length()) {
        return (
            0 ==
            full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

PlanarInit::PlanarInit(bool v) {
    verbose = v;
    // init parameters
    num_images = 0;
    params = PatchMatchParams();
}

PlanarInit::~PlanarInit() {
    // release host memory
    if (sizeof(plane_hypotheses_host) > 0) {
        delete[] plane_hypotheses_host;
        delete[] costs_host;
    }
    // delete[] plane_hypotheses_host;
    // delete[] costs_host;

    // release cuda memory
    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }

    if (params.planar_prior) {
        if (sizeof(prior_planes_host) > 0) {
            delete[] prior_planes_host;
            delete[] plane_masks_host;
        }
        // delete[] prior_planes_host;
        // delete[] plane_masks_host;

        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }
}

void PlanarInit::add_image(const std::string image_path) {
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    ++num_images;
}

void PlanarInit::add_superpixel(const cv::Mat& _superpixel) {
    superpixel = _superpixel;
}

cv::Mat PlanarInit::get_image(const int32_t index) {
    return images[index];
}

void PlanarInit::add_camera(
    vector<float> K,
    vector<float> R,
    vector<float> t,
    int height,
    int width,
    float depth_min,
    float depth_max) {
    Camera cam = load_camera(K, R, t, height, width, depth_min, depth_max);
    cameras.push_back(cam);
}

int32_t PlanarInit::get_num_images() {
    return num_images;
}

Camera PlanarInit::get_camera(const int32_t index) {
    return cameras[index];
}

float PlanarInit::get_cost(const int32_t index) {
    return costs_host[index];
}

void PlanarInit::export_plane(cv::Mat& depth, cv::Mat& normal, cv::Mat& cost) {
    const int32_t cols = cameras[0].width;
    const int32_t rows = cameras[0].height;
    for (int32_t col = 0; col < cols; ++col) {
        for (int32_t row = 0; row < rows; ++row) {
            int32_t index = row * cols + col;
            float4 plane_hypothesis = plane_hypotheses_host[index];
            depth.at<float>(row, col) = plane_hypothesis.w;
            normal.at<cv::Vec3f>(row, col) =
                cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            cost.at<float>(row, col) = costs_host[index];
        }
    }
}

void PlanarInit::cuda_space_init() {
    cudaSetDevice(0);
    if (verbose) {
        std::cout << "num_images: " << num_images << std::endl;
    }

    for (int32_t i = 0; i < num_images; ++i) {
        int32_t rows = images[i].rows;
        int32_t cols = images[i].cols;

        // allocate the cuda memory(cols * rows float32-type) and copy
        // the image data from the host memory to the cuda memory
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);

        cudaMemcpy2DToArray(
            cuArray[i],
            0,
            0,
            images[i].ptr<float>(),
            images[i].step[0],
            cols * sizeof(float),
            rows,
            cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    // malloc the cuda memory and store the images as texture_objects_cuda
    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(
        texture_objects_cuda,
        &texture_objects_host,
        sizeof(cudaTextureObjects),
        cudaMemcpyHostToDevice);

    // malloc the cuda memory and store the camera parameters as cameras_cuda
    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    /* allocate the memory to store the pixelwise plane hypotheses parameters
       (float4: origin_distance + normal)
    */
    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc(
        (void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    // allocate the memory to store the pixelwise costs
    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    // allocate the memory to store the pixelwise random states and selected
    // views
    cudaMalloc(
        (void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc(
        (void**)&selected_views_cuda,
        sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    // allocate the memory to store the output depth
    cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
}

void PlanarInit::cuda_planar_prior_init() {
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc(
        (void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)mask_tri.at<float>(j, i);
            if (mask_tri.at<float>(j, i) > 0) {
                prior_planes_host[center] = planeParams_tri[mask_tri.at<float>(j, i) - 1];
            }
        }
    }

    cudaMemcpy(
        prior_planes_cuda,
        prior_planes_host,
        sizeof(float4) * (cameras[0].height * cameras[0].width),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        plane_masks_cuda,
        plane_masks_host,
        sizeof(unsigned int) * (cameras[0].height * cameras[0].width),
        cudaMemcpyHostToDevice);
}

vector<cv::Point> PlanarInit::get_support_points() {
    const int32_t cols = cameras[0].width;
    const int32_t rows = cameras[0].height;

    vector<cv::Point> support2DPoints;

    // get support points
    support2DPoints.clear();
    const int32_t step_size = 5;
    for (int col = 0; col < cols; col += step_size) {
        for (int row = 0; row < rows; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(cols, col + step_size);
            int r_bound = std::min(rows, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    int center = r * cols + c;
                    if (get_cost(center) < 2.0f && min_cost > get_cost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost = get_cost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }

    return support2DPoints;
}

void PlanarInit::vis_triangulation(
    cv::Rect imageRC,
    vector<Triangle> triangles,
    const std::string triangulation_path) {
    cv::Mat ref_image = images[0];
    vector<cv::Mat> mbgr(3);
    mbgr[0] = ref_image.clone();
    mbgr[1] = ref_image.clone();
    mbgr[2] = ref_image.clone();
    cv::Mat src_image;
    cv::merge(mbgr, src_image);

    for (Triangle triangle : triangles) {
        if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
            imageRC.contains(triangle.pt3)) {
            cv::line(src_image, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
            cv::line(src_image, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
            cv::line(src_image, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
        }
    }
    cv::imwrite(triangulation_path, src_image);
}

float PlanarInit::get_depth_from_plane_param(
    const float4 plane_hypothesis,
    const int x,
    const int y) {
    return -plane_hypothesis.w /
        ((x - cameras[0].K[2]) * plane_hypothesis.x / cameras[0].K[0] +
         (y - cameras[0].K[5]) * plane_hypothesis.y / cameras[0].K[4] + plane_hypothesis.z);
}

void PlanarInit::process_mask(
    const cv::Mat& depth,
    const cv::Rect imageRC,
    vector<Triangle> triangles) {
    const int32_t cols = cameras[0].width;
    const int32_t rows = cameras[0].height;
    mask_tri = cv::Mat::zeros(rows, cols, CV_32FC1);
    planeParams_tri.clear();

    uint32_t idx = 0;
    for (const auto triangle : triangles) {
        if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) &&
            imageRC.contains(triangle.pt3)) {
            float L01 = sqrt(
                pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
            float L02 = sqrt(
                pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
            float L12 = sqrt(
                pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

            float max_edge_length = std::max(L01, std::max(L02, L12));
            float step = 1.0 / max_edge_length;

            for (float p = 0; p < 1.0; p += step) {
                for (float q = 0; q < 1.0 - p; q += step) {
                    int x =
                        p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                    int y =
                        p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                    mask_tri.at<float>(y, x) = idx + 1.0; // To distinguish from the label
                                                          // of non-triangulated areas
                }
            }

            // estimate plane parameter
            float4 n4 = get_prior_plane_params(triangle, depth);
            planeParams_tri.push_back(n4);
            idx++;
        }
    }

    cv::Mat prior_depths = cv::Mat::zeros(rows, cols, CV_32FC1);
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            if (mask_tri.at<float>(j, i) > 0) {
                float d =
                    get_depth_from_plane_param(planeParams_tri[mask_tri.at<float>(j, i) - 1], i, j);
                if (d <= params.depth_max && d >= params.depth_min) {
                    prior_depths.at<float>(j, i) = d;
                } else {
                    mask_tri.at<float>(j, i) = 0;
                }
            }
        }
    }
}

void PlanarInit::triangulation(const cv::Mat& depth, const std::string triangulation_path = " ") {
    const int32_t cols = cameras[0].width;
    const int32_t rows = cameras[0].height;

    // get the support points
    const vector<cv::Point> support2DPoints = get_support_points();

    vector<Triangle> triangles;
    const cv::Rect imageRC(0, 0, cols, rows);
    cv::Subdiv2D subdiv2d(imageRC);
    for (const auto point : support2DPoints) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    vector<cv::Vec6f> temp_triangles;
    subdiv2d.getTriangleList(temp_triangles);

    const float zero_ratio_thresh = 0.1;
    for (const auto temp_vec : temp_triangles) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        // triangles.push_back(Triangle(pt1, pt2, pt3));
        bool retain = true;
        if (params.superpixel_filter) {
            float L01 = sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
            float L02 = sqrt(pow(pt1.x - pt3.x, 2) + pow(pt1.y - pt3.y, 2));
            float L12 = sqrt(pow(pt2.x - pt3.x, 2) + pow(pt2.y - pt3.y, 2));
            // filter out the triangles cover multiple superpixels
            float max_edge_length = std::max(L01, std::max(L02, L12));
            float step = 1.0 / max_edge_length;

            float count = 0.0f;
            float zero_count = 0.0f;
            uint8_t seg_id = 0;
            for (float p = 0; (p < 1.0) && retain; p += step) {
                for (float q = 0; (q < 1.0 - p) && retain; q += step) {
                    int32_t x = p * pt1.x + q * pt2.x + (1.0 - p - q) * pt3.x;
                    int32_t y = p * pt1.y + q * pt2.y + (1.0 - p - q) * pt3.y;
                    const uint8_t s_id = superpixel.at<uint8_t>(y, x);
                    count += 1.0f;
                    if (s_id == 0) {
                        zero_count += 1.0f;
                    }
                    if (seg_id == 0) {
                        seg_id = s_id;
                    } else if (s_id == 0) {
                        continue;
                    } else if (s_id != seg_id) {
                        // std::cout << "s_id: (" << static_cast<int32_t>(s_id) << ") seg_id: ("
                        //           << static_cast<int32_t>(seg_id) << ")" << std::endl;
                        retain = false;
                    }
                }
            }
            const float zero_ratio = zero_count / count;
            if (zero_ratio > zero_ratio_thresh) {
                // std::cout << "zero_ratio: (" << zero_count << "/" << count << ") - " << zero_ratio << std::endl;
                retain = false;
            }
        }
        if (retain) {
            triangles.push_back(Triangle(pt1, pt2, pt3));
        }
    }

    if (has_ending(triangulation_path, ".png")) {
        if (verbose) {
            std::cout << "save triangulation as " << triangulation_path << std::endl;
        }
        vis_triangulation(imageRC, triangles, triangulation_path);
    }

    process_mask(depth, imageRC, triangles);
}

__global__ void sliding_window_normal_kernel(
    const int32_t* __restrict__ valid_normal_ptr,
    const float* __restrict__ input_normal_ptr,
    const int32_t height,
    const int32_t width,
    const int32_t window_size,
    // output
    float* __restrict__ output_normal_ptr
) {
    const int32_t row = blockIdx.x;
    const int32_t col = threadIdx.x;

    const int32_t half_size = window_size / 2;
    // judge the planar region
    if (valid_normal_ptr[row * width + col] == 1) {
        output_normal_ptr[(row * width + col) * 3 + 0] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 1] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 2] = 0.f;
        return;
    }
    // sliding windows
    int32_t valid_count = 0;
    float3 n = make_float3(0.f, 0.f, 0.f);
    for (int32_t r = row - half_size; r < row + half_size + 1; ++r) {
        for (int32_t c = col - half_size; c < col + half_size + 1; ++c) {
            if (r < 0 || r >= height || c < 0 || c >= width) {
                continue;
            }
            if (valid_normal_ptr[r * width + c] == 1) {
                continue;
            }
            ++valid_count;
            n.x += input_normal_ptr[(r * width + c) * 3 + 0];
            n.y += input_normal_ptr[(r * width + c) * 3 + 1];
            n.z += input_normal_ptr[(r * width + c) * 3 + 2];
        }
    }

    n.x /= valid_count;
    n.y /= valid_count;
    n.z /= valid_count;
    __syncthreads();

    // enough valid normal
    output_normal_ptr[(row * width + col) * 3 + 0] = n.x;
    output_normal_ptr[(row * width + col) * 3 + 1] = n.y;
    output_normal_ptr[(row * width + col) * 3 + 2] = n.z;
}

cv::Mat sliding_window_normal(
    const cv::Mat& planar_region,
    const cv::Mat& normal,
    const int32_t window_size) {
    const int32_t height = normal.rows;
    const int32_t width = normal.cols;
    cv::Mat smooth_normal = cv::Mat::zeros(height, width, CV_32FC3);

    // cuda pointer
    // bool* cuda_planar_region_ptr = NULL;
    // float* cuda_normal_ptr = NULL;
    // float* cuda_smooth_normal_ptr = NULL;
    // size_t normal_size = height * width * 3 * sizeof(float);

    const int32_t half_size = window_size / 2;
    for (int32_t row = 0; row < height; ++row) {
        for (int32_t col = 0; col < width; ++col) {
            // judge the planar region
            if (planar_region.at<int32_t>(row, col) == 1) {
                smooth_normal.at<cv::Vec3f>(row, col) = cv::Vec3f(0.0f, 0.0f, 0.0f);
                continue;
            }
            // sliding windows
            int32_t valid_count = 0;
            cv::Vec3f n(0.0f, 0.0f, 0.0f);
            for (int32_t r = row - half_size; r < row + half_size + 1; ++r) {
                for (int32_t c = col - half_size; c < col + half_size + 1; ++c) {
                    if (r < 0 || r >= height || c < 0 || c >= width) {
                        continue;
                    }
                    if (planar_region.at<int32_t>(r, c) == 1) {
                        continue;
                    }
                    ++valid_count;
                    n += normal.at<cv::Vec3f>(r, c);
                }
            }
            n /= valid_count;
            smooth_normal.at<cv::Vec3f>(row, col) = n;
        }
    }

    // cudaMalloc((void**)&cuda_planar_region_ptr, height * width * sizeof(int32_t));
    // cudaMalloc((void**)&cuda_normal_ptr, normal_size);
    // cudaMalloc((void**)&cuda_smooth_normal_ptr, normal_size);
    // cudaMemcpy(cuda_planar_region_ptr, planar_region.ptr<int32_t>(), height * width * sizeof(int32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(cuda_normal_ptr, normal.ptr<float>(), normal_size, cudaMemcpyHostToDevice);

    // sliding_window_normal_kernel<<<height, width>>>(
    //     cuda_planar_region_ptr,
    //     cuda_normal_ptr,
    //     height,
    //     width,
    //     window_size,
    //     cuda_smooth_normal_ptr
    // );
    
    // cudaMemcpy(smooth_normal.ptr<float>(), cuda_smooth_normal_ptr, normal_size, cudaMemcpyDeviceToHost);
    // cudaFree(cuda_planar_region_ptr);
    // cudaFree(cuda_normal_ptr);
    // cudaFree(cuda_smooth_normal_ptr);

    // return normal;
    return smooth_normal;
}

cv::Mat fusion_planar_mask(
    const cv::Mat& seg_ids,
    const cv::Mat& planar_mask,
    const cv::Mat& filter_normals,
    const int32_t rows,
    const int32_t cols,
    const int32_t thresh = 100,
    const float nonplanar_percent = 0.75,
    // matching paramters
    const float cos_sim_thresh = 0.8f,
    const float match_ratio_thresh = 0.9f) {
    double min_val, max_val;
    cv::minMaxLoc(seg_ids, &min_val, &max_val);

    vector<vector<cv::Vec3f>> seg_normals;
    vector<vector<cv::Point2i>> seg_ids_map;
    vector<cv::Point2i> seg_ids_planar_count;
    seg_normals.resize(static_cast<uint32_t>(max_val) + 1);
    seg_ids_map.resize(static_cast<uint32_t>(max_val) + 1);
    seg_ids_planar_count.resize(static_cast<uint32_t>(max_val) + 1);

    // fullfill the seg_ids_map
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const cv::Point2i p(col, row);
            const int32_t seg_id = seg_ids.at<int32_t>(p);
            seg_ids_map[seg_id].push_back(p);
            if (planar_mask.at<int32_t>(p) == 0) { // nonplanar region
                ++seg_ids_planar_count[seg_id].x;
                // statistic normals
                seg_normals[seg_id].push_back(filter_normals.at<cv::Vec3f>(p));
            } else { // planar region
                ++seg_ids_planar_count[seg_id].y;
            }
        }
    }

    uint32_t vpsarea_seg_id = 301; // too textureless region, where mvs could not handle, segid begin from 301
    cv::Mat fusion_mask = cv::Mat::zeros(rows, cols, CV_32SC1);
    for (int32_t seg_id = 1; seg_id < static_cast<uint32_t>(max_val) + 1; ++seg_id) {
        const float nonplanar = static_cast<float>(seg_ids_planar_count[seg_id].x);
        const float planar = static_cast<float>(seg_ids_planar_count[seg_id].y);
        const float amount = nonplanar + planar;
        // filter out the small superpixels
        // filter out the small, useless(too many nonplanar) superpixels
        if (amount <= thresh) {
            continue;
        }
        bool skip = false;
        // filter out the truly nonplanar superpixels
        if ((nonplanar / amount) >= nonplanar_percent) {
            skip = true;
            // calculate the average normal
            cv::Vec3f avg_n(0.0f, 0.0f, 0.0f);
            for (cv::Vec3f n : seg_normals[seg_id]) {
                avg_n += n;
            }
            avg_n /= nonplanar;

            // count the number of the normals that match the average normal
            int32_t valid = 0;
            for (cv::Vec3f n : seg_normals[seg_id]) {
                const float cos_sim = avg_n.dot(n);
                valid += (cos_sim > cos_sim_thresh ? 1 : 0);
            }
            // define the superpixel as planar superpixel if there are enough normals matching the
            // average normal, and vice versa
            const float ratio = static_cast<float>(valid) / nonplanar;
            if (ratio > match_ratio_thresh) {  // too many planar
                skip = false;
            } else {                           // really object
                for (cv::Point2i p : seg_ids_map[seg_id]) {
                    fusion_mask.at<int32_t>(p) = 300;
                }
            }
        } else if ((nonplanar / amount) < 0.25){    // too textureless region, where mvs could not handle
            // keep this region
            for (cv::Point2i p : seg_ids_map[seg_id]) {
                fusion_mask.at<int32_t>(p) = vpsarea_seg_id;
            }
            vpsarea_seg_id += 1;
            skip = true;
        } else {
            skip = false;
        }
        // skip the superpixel whose normals are not quite consistent(not planar)
        if (skip) {
            continue;
        }
        for (cv::Point2i p : seg_ids_map[seg_id]) {
            fusion_mask.at<int32_t>(p) = seg_id;
        }
    }

    return fusion_mask;
}

cv::Mat filter_by_var_map(
    const cv::Mat& var_map,
    const cv::Mat& segment_ids_map,
    const float var_thresh) {
    
    const int32_t rows = var_map.rows, cols = var_map.cols;
    cv::Mat output_segment_ids_map = cv::Mat::zeros(rows, cols, CV_32SC1);
    
    double min_val, max_val;
    cv::minMaxLoc(segment_ids_map, &min_val, &max_val);
    vector<float> var_accum;
    vector<float> count;
    vector<bool> filter_mask;
    const int32_t max_seg_id = static_cast<int32_t>(max_val);
    var_accum.resize(max_seg_id + 1);
    count.resize(max_seg_id + 1);
    filter_mask.resize(max_seg_id + 1);

    for (int32_t seg_id = 1; seg_id < max_seg_id; ++seg_id) {
        var_accum[seg_id] = 0;
        count[seg_id] = 0;
        filter_mask[seg_id] = false;
    }
    // statistic
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int32_t seg_id = segment_ids_map.at<int32_t>(row, col);
            if (seg_id == 0) continue;
            ++count[seg_id];
            var_accum[seg_id] += var_map.at<float>(row, col);
        }
    }
    
    // calculate the average var and fileter
    for (int32_t seg_id = 1; seg_id < max_seg_id + 1; ++seg_id) {
        const float mean_var = var_accum[seg_id] / static_cast<float>(count[seg_id]);
        filter_mask[seg_id] = mean_var <= var_thresh;
    }
    
    // filter
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int32_t seg_id = segment_ids_map.at<int32_t>(row, col);
            if (filter_mask[seg_id]) {
                output_segment_ids_map.at<int32_t>(row, col) = seg_id;
            }
        }
    }

    return output_segment_ids_map;
}

float3 PlanarInit::get_3D_point_on_ref_cam(
    const int32_t x,
    const int32_t y,
    const float depth,
    const Camera camera) {
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}

float4 PlanarInit::get_prior_plane_params(const Triangle triangle, const cv::Mat depth) {
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = get_3D_point_on_ref_cam(
        triangle.pt1.x,
        triangle.pt1.y,
        depth.at<float>(triangle.pt1.y, triangle.pt1.x),
        cameras[0]);
    float3 ptX2 = get_3D_point_on_ref_cam(
        triangle.pt2.x,
        triangle.pt2.y,
        depth.at<float>(triangle.pt2.y, triangle.pt2.x),
        cameras[0]);
    float3 ptX3 = get_3D_point_on_ref_cam(
        triangle.pt3.x,
        triangle.pt3.y,
        depth.at<float>(triangle.pt3.y, triangle.pt3.x),
        cameras[0]);

    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 =
        make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}
