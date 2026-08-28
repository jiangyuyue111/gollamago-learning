from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent

setup(
    name="maca-kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="operators.maca_cpp.maca_kernels",
            sources=[
                str(ROOT / "src" / "bindings.cpp"),
                str(ROOT / "src" / "kernels.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
