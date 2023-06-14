\name{print-Rcpp_MMPR}
\alias{print-Rcpp_MMPR}
\alias{print,Rcpp_MMPR-method}
\alias{show,Rcpp_MMPR-method}
\title{print-Rcpp_MMPR}
\usage{
\S4method{print}{Rcpp_MMPR}(x)

\S4method{show}{Rcpp_MMPR}(object)
}
\arguments{
\item{x, object}{A MMPR vector/matrix.}

}
\description{
show function will print the precision and type of the object, and print will print the meta
data of the object without printing the values. function (x$PrintValues() should be used to print the values)
}
\details{
prints metadata about the object and some values.
}
\examples{
  \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    y <- as.MMPR(1:20,4,5,"double")
    x
    print(y)
    }
}
