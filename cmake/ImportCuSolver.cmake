set(cudart_lib CUDA::cudart_static)
set(cublas_lib CUDA::cublas_static)
set(LIBS
        CUDA::cusolver_static
        CUDA::cublas_static
        CUDA::cublasLt_static
        CUDA::cudart_static
        ${LIBS}
        )