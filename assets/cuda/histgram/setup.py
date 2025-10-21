from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12'],
    'nvcc': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12'],
}

setup(
    name='histgram',
    ext_modules=[
        CUDAExtension(
            name='histgram',
            sources=[
                "/".join(__file__.split('/')[:-1] + ['histgram_cuda.cpp']), # must named as xxx_cuda.cpp
                "/".join(__file__.split('/')[:-1] + ['histgram.cu']),
            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.0'
)