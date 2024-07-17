
# MPCR: Multi-Precision Computing in R

## Overview

The **MPCR** package provides new data-structure support for multi- and mixed-precision for R users, supporting 16-bit, 32-bit, and 64-bit operations.
This enables optimized memory allocation based on the desired precision, offering significant advantages in-memory optimization and computational efficiency.
In addition, **MPCR** leverages GPU acceleration through CUDA, allowing for high-performance computations on the GPU.
This capability includes seamless memory transfers between CPU and GPU, managed automatically by the package, simplifying the setup and usage for end users.

## Key Features

### 1. Multi-Precision Support

**MPCR** introduces a new data structure that supports three different precisions:
- **16-bit** - Supported on GPU only, with half-precision support for the Matrix-Matrix Multiplication only ( crossprod () ).
- **32-bit**
- **64-bit**

This flexibility allows for optimized memory allocation and performance tuning based on the specific needs of your application.

### 2. Comprehensive Linear Algebra Support

The package extends support to all basic linear algebra methods across different precisions. You can perform operations like matrix multiplication, inversion, and eigenvalue decomposition with ease, regardless of the chosen precision.

### 3. Seamless Integration

**MPCR** maintains a consistent interface with normal R functions, allowing for seamless code integration. You can use MPCR data structures and functions without having to significantly alter your existing codebase, ensuring a user-friendly experience.

### 4. GPU Acceleration with CUDA

The package now supports CUDA, enabling the allocation of memory and performance of operations on the GPU using CuSolver and CuBLAS. This includes:
- Automatic handling of data transfers between CPU and GPU.
- Simple selection mechanism for performing operations on either CPU or GPU, with MPCR managing the complexities behind the scenes.

## 5. Installation

To install the **MPCR** package, you can use the following command:

- Clone the MPCR package from [here](https://github.com/stsds/MPCR).
- Checkout to tag `v1.0.0` (Latest stable release).
- Run `R CMD INSTALL .` from the project root directory.
___


## 6. Requirements
- Rcpp (needs to be installed before trying to install the package), use `install.packages("Rcpp")` in R to install it.
- For optimal performance, `MKL` is recommended for building the package,
in case MKL is not found on the system, the package will automatically download [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS).
- [Blaspp](https://github.com/icl-utk-edu/blaspp) will be installed automatically.
- [Lapackpp](https://github.com/icl-utk-edu/lapackpp) will be installed automatically.
- [CUDA ToolKit > 11.4](https://developer.nvidia.com/cuda-toolkit) in case CUDA toolkit is not available on the system, the package will be installed with CPU support only.
___

## 7. Getting Started

Example showcasing how MPCR can be used to allocate and perform operations on CPU/GPU. More examples are available
in the following [directory](tests/R-tests).

```R
library("MPCR")

values <- c(3.12393, -1.16854, -0.304408, -2.15901,
            -1.16854, 1.86968, 1.04094, 1.35925, -0.304408,
            1.04094, 4.43374, 1.21072, -2.15901, 1.35925, 1.21072, 5.57265)

# placement will indicate on which device the allocation should be made.
# default option : CPU

A <- as.MPCR(values, nrow = 4, ncol = 4, precision = "float",placement="GPU")
B <- as.MPCR(values, nrow = 4, ncol = 4, precision = "float",placement="GPU")

# Set operation placement is the function used to decide whether the up-coming
# operation should be executed on CPU or GPU


# All the up-coming operation will be executed on GPU
# default option : CPU
MPCR.SetOperationPlacement("GPU")

cat("----------------------- CrossProduct C=XY --------------------\n")
crossproduct$PrintValues()
# Print is a CPU only function, so the data will automatically be transferred to CPU,
# resulting in having a copy on CPU and a copy on GPU for future GPU operation
crossproduct <- crossprod(x, y)
```