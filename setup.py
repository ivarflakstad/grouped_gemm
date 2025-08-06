from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Check if ROCm is available
def is_rocm_available():
    return torch.version.hip is not None and torch.cuda.is_available()

# Get ROCm/HIP paths
def get_rocm_paths():
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    hip_path = os.environ.get('HIP_PATH', f'{rocm_path}/hip')
    return rocm_path, hip_path

rocm_path, hip_path = get_rocm_paths()

# Library directories
library_dirs = [
    f'{rocm_path}/lib',
    f'{hip_path}/lib',
    f'{rocm_path}/lib64'
]

# Compiler and linker flags for ROCm/HIP
extra_compile_args = {
    'cxx': [
        '-O1',
        '-std=c++17',
        '-fPIC',
        f'-I{rocm_path}/include',
        f'-I{hip_path}/include',
        f'-I{rocm_path}/include/ck',
        '-D__HIP_PLATFORM_AMD__',
        '-DUSE_ROCM',
        '-U__HIP_NO_HALF_CONVERSIONS__',
        '-U__HIP_NO_HALF_OPERATORS__'
    ],
    'nvcc': [
        '-O1',
        '--std=c++17',
        f'-I{rocm_path}/include',
        f'-I{hip_path}/include',
        f'-I{rocm_path}/include/ck',
        '-D__HIP_PLATFORM_AMD__',
        '-DUSE_ROCM',
        '--offload-arch=gfx942',
        '-U__HIP_NO_HALF_CONVERSIONS__',
        '-U__HIP_NO_HALF_OPERATORS__'
    ]
}

setup(
    name='grouped_gemm',
    ext_modules=[
        CUDAExtension('grouped_gemm_backend', [
            'csrc/grouped_gemm.cu',
        ],
        include_dirs=[
            f'{rocm_path}/include',
            f'{hip_path}/include',
            f'{rocm_path}/include/ck',
            f'{rocm_path}/include/ck_tile',
        ] + torch.utils.cpp_extension.include_paths(),
        #libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=[
            #f'-L{rocm_path}/lib',
            #f'-L{rocm_path}/lib64',
            #f'-L{hip_path}/lib',
            #'-Wl,-rpath,' + f'{rocm_path}/lib',
            #'-Wl,-rpath,' + f'{rocm_path}/lib64'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
