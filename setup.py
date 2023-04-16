# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import warnings
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

__version__ = None
exec(open("gmvs/version.py", "r").read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []
include_dirs = [os.path.join(ROOT_DIR, "gmvs", "include")]

try:
    ext_modules = [
        CUDAExtension(
            name="gmvs.src",
            sources=[
                "gmvs/src/bindings.cpp",
                "gmvs/src/data_structure.cpp",
                "gmvs/src/fusion.cu",
                "gmvs/src/patch_match.cu",
            ],
            include_dirs=include_dirs,
            optional=False,
        ),
    ]
except:
    import warnings

    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

# build the lib
os.system("cd gmvs/lib && sh build.sh")

setup(
    name="gmvs",
    version=__version__,
    author="Zhihao Liang",
    author_email="eezhihaoliang@mail.scut.edu.cn",
    description="Multi View Stereo of Gorilla-Lab",
    long_description="Multi View Stereo of Gorilla-Lab",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0"],
    packages=["gmvs", "gmvs.src"],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
