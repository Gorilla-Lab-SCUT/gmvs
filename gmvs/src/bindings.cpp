// Copyright 2022 Gorilla-Lab
#include <torch/extension.h>

#include "data_structure.hpp"
#include "fusion.cuh"
#include "patch_match.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_camera", &mvs::load_camera);
    m.def("load_problem", &mvs::load_problem);

    py::class_<mvs::Fuser>(m, "Fuser")
        .def(py::init<>())
        .def("load_samples", &mvs::Fuser::load_samples)
        .def("run_fusion", &mvs::Fuser::run_fusion)
        .def_readwrite("depths_host", &mvs::Fuser::depths_host)
        .def_readwrite("normals_host", &mvs::Fuser::normals_host)
        .def_readwrite("masks_cuda", &mvs::Fuser::masks_cuda);

    py::class_<mvs::PatchMatcher>(m, "PatchMatcher")
        .def(py::init<>())
        .def("add_samples", &mvs::PatchMatcher::add_samples)
        .def("run_patch_match", &mvs::PatchMatcher::run_patch_match)
        .def_readwrite("params", &mvs::PatchMatcher::params);

    py::class_<mvs::PatchMatchParams>(m, "PatchMatchParams")
        .def(py::init<>())
        .def_readwrite("max_iterations", &mvs::PatchMatchParams::max_iterations)
        .def_readwrite("patch_size", &mvs::PatchMatchParams::patch_size)
        .def_readwrite("num_images", &mvs::PatchMatchParams::num_images)
        .def_readwrite("max_image_size", &mvs::PatchMatchParams::max_image_size)
        .def_readwrite("radius_increment", &mvs::PatchMatchParams::radius_increment)
        .def_readwrite("sigma_spatial", &mvs::PatchMatchParams::sigma_spatial)
        .def_readwrite("sigma_color", &mvs::PatchMatchParams::sigma_color)
        .def_readwrite("top_k", &mvs::PatchMatchParams::top_k)
        .def_readwrite("baseline", &mvs::PatchMatchParams::baseline)
        .def_readwrite("depth_min", &mvs::PatchMatchParams::depth_min)
        .def_readwrite("depth_max", &mvs::PatchMatchParams::depth_max)
        .def_readwrite("disparity_min", &mvs::PatchMatchParams::disparity_min)
        .def_readwrite("disparity_max", &mvs::PatchMatchParams::disparity_max)
        .def_readwrite("geom_consistency", &mvs::PatchMatchParams::geom_consistency)
        .def_readwrite("multi_geometry", &mvs::PatchMatchParams::multi_geometry)
        .def_readwrite("planar_prior", &mvs::PatchMatchParams::planar_prior);

    py::class_<mvs::Problem>(m, "Problem")
        .def(py::init<>())
        .def_readwrite("ref_image_id", &mvs::Problem::ref_image_id);

    py::class_<mvs::Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("height", &mvs::Camera::height)
        .def_readwrite("width", &mvs::Camera::width)
        .def_readwrite("depth_min", &mvs::Camera::depth_min)
        .def_readwrite("depth_max", &mvs::Camera::depth_max);
}
