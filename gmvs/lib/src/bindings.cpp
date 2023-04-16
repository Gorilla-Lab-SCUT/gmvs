// Copyright 2022 Gorilla-Lab
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray_converter.h"
#include "planar_initer.cuh"
#include "planar_initer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(planar_initer, m) {
    NDArrayConverter::init_numpy();

    m.def("load_camera", &load_camera);
    m.def("load_problem", &load_problem);
    m.def("fusion_planar_mask", &fusion_planar_mask);
    m.def("filter_by_var_map", &filter_by_var_map);
    m.def("sliding_window_normal", &sliding_window_normal);

    py::class_<PatchMatchParams>(m, "PatchMatchParams")
            .def(py::init<>())
            .def_readwrite("max_iterations", &PatchMatchParams::max_iterations)
            .def_readwrite("patch_size", &PatchMatchParams::patch_size)
            .def_readwrite("num_images", &PatchMatchParams::num_images)
            .def_readwrite("max_image_size", &PatchMatchParams::max_image_size)
            .def_readwrite("radius_increment", &PatchMatchParams::radius_increment)
            .def_readwrite("sigma_spatial", &PatchMatchParams::sigma_spatial)
            .def_readwrite("sigma_color", &PatchMatchParams::sigma_color)
            .def_readwrite("top_k", &PatchMatchParams::top_k)
            .def_readwrite("baseline", &PatchMatchParams::baseline)
            .def_readwrite("depth_min", &PatchMatchParams::depth_min)
            .def_readwrite("depth_max", &PatchMatchParams::depth_max)
            .def_readwrite("disparity_min", &PatchMatchParams::disparity_min)
            .def_readwrite("disparity_max", &PatchMatchParams::disparity_max)
            .def_readwrite("geom_consistency", &PatchMatchParams::geom_consistency)
            .def_readwrite("multi_geometry", &PatchMatchParams::multi_geometry)
            .def_readwrite("planar_prior", &PatchMatchParams::planar_prior)
            .def_readwrite("superpixel_filter", &PatchMatchParams::superpixel_filter);

    py::class_<Problem>(m, "Problem")
            .def(py::init<>())
            .def_readwrite("ref_image_id", &Problem::ref_image_id);

    py::class_<Camera>(m, "Camera")
            .def(py::init<>())
            .def_readwrite("height", &Camera::height)
            .def_readwrite("width", &Camera::width)
            .def_readwrite("depth_min", &Camera::depth_min)
            .def_readwrite("depth_max", &Camera::depth_max);

    py::class_<PlanarInit>(m, "PlanarInit")
            .def(py::init<const bool>())
            .def("add_superpixel", &PlanarInit::add_superpixel)
            .def("add_image", &PlanarInit::add_image)
            .def("get_image", &PlanarInit::get_image)
            .def("add_camera", &PlanarInit::add_camera)
            .def("get_camera", &PlanarInit::get_camera)
            .def("get_num_images", &PlanarInit::get_num_images)
            .def("cuda_space_init", &PlanarInit::cuda_space_init)
            .def("cuda_planar_prior_init", &PlanarInit::cuda_planar_prior_init)
            .def("run_patch_match", &PlanarInit::run_patch_match)
            .def("jump_patch_match", &PlanarInit::jump_patch_match)
            .def("export_plane", &PlanarInit::export_plane)
            .def("triangulation", &PlanarInit::triangulation)
            .def_readwrite("params", &PlanarInit::params);
}
