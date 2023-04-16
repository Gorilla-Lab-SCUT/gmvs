#include "planar_initer.hpp"

Camera load_camera(vector<float> K,
                   vector<float> R,
                   vector<float> t,
                   int32_t height,
                   int32_t width,
                   float depth_min,
                   float depth_max) {
    assert(K.size() == 9);
    assert(R.size() == 9);
    assert(t.size() == 3);
    Camera cam;

    for (int32_t i = 0; i < 9; ++i) {
        cam.K[i] = K[i];
        cam.R[i] = R[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        cam.t[i] = t[i];
    }
    cam.height = height;
    cam.width = width;
    cam.depth_min = depth_min;
    cam.depth_max = depth_max;

    return cam;
}

Problem load_problem(int32_t ref_id, vector<int32_t> src_ids) {
    Problem problem;

    problem.ref_image_id = ref_id;
    assert(src_ids.size() <= MAX_NGB);
    problem.num_ngb = src_ids.size();
    std::copy(src_ids.begin(), src_ids.end(), problem.src_image_ids);
    return problem;
}
