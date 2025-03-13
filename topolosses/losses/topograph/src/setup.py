from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sysconfig

__version__ = "1.0.0"
conda_prefix = os.getenv("CONDA_PREFIX")
print(conda_prefix)
extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
# eigen_path = os.path.abspath(os.path.join(".", "eigen4"))
# print("Eigen path:", eigen_path)

ext_modules = [
    Pybind11Extension(
        "Topograph",
        sorted(glob("ext/*.cpp")), 
        include_dirs=[
            os.path.join(conda_prefix, "include", "opencv4"),
            os.path.join(conda_prefix, "include", "eigen3"),
            os.path.join(conda_prefix, "include", "boost"),
        ],
        library_dirs=[os.path.join(conda_prefix, "lib")],
        libraries=["opencv_core", "opencv_imgproc"],
        extra_compile_args=["-fopenmp"],
    ),
]

setup(
    name="Topograph",
    version=__version__,
    author="Alexander H. Berger",
    author_email="a.berger@tum.de",
    url="",
    description="Highly Efficient C++ Topograph implementation",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
