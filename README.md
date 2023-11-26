
# MPCR

MPCR is an advanced package designed to provide R users with a customized data structure.
This package harnesses the combined strength of C++ and R, empowering users with high-performance computing capabilities.
Specifically tailored for researchers and data scientists working with multi or mixed-precision arithmetic,
MPCR serves as an invaluable tool for achieving efficient and accurate computations.

##### MPCR provides R users with a primary, tailor-made data structures:
- Normal matrix/vector with different precision allocation (16-bit (half-precision), 32-bit (single-precision), and 64-bit (double precision)).
___

## Requirements
- Rcpp (needs to be installed before trying to install the package), use `install.packages("Rcpp")` in R to install it.
- For optimal performance, `MKL` is recommended for building the package,
in case MKL is not found on the system, the package will automatically download [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS).
- [Blaspp](https://github.com/icl-utk-edu/blaspp) (if not found, it will be installed automatically).
- [Lapackpp](https://github.com/icl-utk-edu/lapackpp) (if not found, it will be installed automatically).

___

## Installation
To install the package:
- Clone the MPCR package from [here](https://github.com/stsds/MPCR).
- Checkout to tag `v1.0.0` (Latest stable release).
- Run `R CMD INSTALL .` from the project root directory.
___