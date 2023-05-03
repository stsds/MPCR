# MPR

Multi-Precision R-package providing an interface for R using Rcpp Modules to use multiple precisions for floating point number representation and arithmetic.


**MPR** is a Multi precision Data Type framework for R.  Base R has no single precision type.  Its "numeric" vectors/matrices are double precision (or possibly integer).
Floats have half the precision of double precision data,and sfloat has half the precision of float (used on NVIDIA) , for a pretty obvious performance vs accuracy tradeoff.

A vector/matrix of floats should use about half as much memory as a matrix of doubles, and your favorite vector/matrix routines will generally compute about twice as fast on them as well.  However, the results will not be as accurate, and are much more prone to roundoff error/mass cancellation issues.  Statisticians have a habit of over-hyping the dangers of roundoff error in this author's opinion.  If your data is [well-conditioned](https://en.wikipedia.org/wiki/Condition_number), then using MPR is fine for many applications, and same goes for sfloat.


## Installation

### Requirements:
- Rcpp
- Blas/lapack

To install the R package from source ,run:

```shell
./InstallScript.sh
```

#### To Run the R package:
- To use the package you will need to add lapackpp and blaspp lib to the LD_LIBRARY_PATH. To do so :
```shell
cd MPR/
source env.sh
```


## Testing

To Test the CPP code ,run:

```shell
source env.sh  #from project dir
./config.sh -t
./clean_build.sh
cd bin/tests/cpp-tests/
./system-tests
```