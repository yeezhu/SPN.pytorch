import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['spn/src/libspn.c']
headers = ['spn/src/libspn.h']
extra_objects = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['spn/src/libspn_cuda.c']
    headers += ['spn/src/libspn_cuda.h']
    extra_objects += ['spn/src/libspn_kernel.cu.o']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'spn._ext.libspn',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    include_dirs=['spn/src'],
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=['-fopenmp'],
)

if __name__ == '__main__':
    ffi.build()
