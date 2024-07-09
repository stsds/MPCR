set(cudart_lib CUDA::cudart)
set(cublas_lib CUDA::cublas)
set(LIBS
        CUDA::cusolver
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart
        ${LIBS}
        )