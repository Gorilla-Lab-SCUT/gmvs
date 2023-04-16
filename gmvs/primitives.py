# Copyright (c) Gorilla-Lab. All rights reserved.
import json
from dataclasses import dataclass
from typing import List, Type


@dataclass
class Problem:
    ref_image_id: int
    src_image_ids: List[int]

    def _to_cpp(self, planar_init: bool = False) -> Type:
        """Generate object to pass to C++

        Returns:
            Problem: the C++ object of rendering options
        """
        if planar_init:
            from gmvs.lib.build import planar_initer

            opt = planar_initer.load_problem(self.ref_image_id, self.src_image_ids)
        else:
            import gmvs.src as _C

            opt = _C.load_problem(self.ref_image_id, self.src_image_ids)

        return opt

    # even the dataclass realize a reliable __repr__, we would like a json style __repr__
    def __repr__(self) -> str:
        content = json.dumps(self.__dict__, indent=4, ensure_ascii=False)
        return content


@dataclass
class PatchMatchParams:
    max_iterations: int = 3
    patch_size: int = 11
    num_images: int = 5
    max_image_size: int = 3200
    radius_increment: int = 2
    sigma_spatial: float = 5.0
    sigma_color: float = 3.0
    top_k: int = 4
    baseline: float = 0.54
    depth_min: float = 0.0
    depth_max: float = 1.0
    disparity_min: float = 0.0
    disparity_max: float = 1.0
    geom_consistency: bool = False
    multi_geometry: bool = False
    planar_prior: bool = False

    def _to_cpp(self) -> Type:
        """Generate object to pass to C++

        Returns:
            PatchMatchParams: the C++ object of rendering options
        """
        from gmvs.lib.build import planar_initer

        opt = planar_initer.PatchMatchParams()
        opt.max_iterations = self.max_iterations
        opt.patch_size = self.patch_size
        opt.num_images = self.num_images
        opt.max_image_size = self.max_image_size
        opt.radius_increment = self.radius_increment
        opt.sigma_spatial = self.sigma_spatial
        opt.sigma_color = self.sigma_color
        opt.top_k = self.top_k
        opt.baseline = self.baseline
        opt.depth_min = self.depth_min
        opt.depth_max = self.depth_max
        opt.disparity_min = self.disparity_min
        opt.disparity_max = self.disparity_max
        opt.geom_consistency = self.geom_consistency
        opt.multi_geometry = self.multi_geometry
        opt.planar_prior = self.planar_prior

        return opt

    # even the dataclass realize a reliable __repr__, we would like a json style __repr__
    def __repr__(self) -> str:
        content = json.dumps(self.__dict__, indent=4, ensure_ascii=False)
        return content
