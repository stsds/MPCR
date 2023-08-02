# Set cuda compilation options.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_ARCHITECTURES "35;50;72")

find_package(CUDAToolkit REQUIRED)
