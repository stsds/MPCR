\name{26-Print}
\alias{print}
\alias{MPCR.str}
\alias{print,Rcpp_MPCR-method}
\alias{show,Rcpp_MPCR-method}

\alias{MPCR.print}
\alias{MPCR.show}

\title{print}
\usage{
\S4method{print}{Rcpp_MPCR}(x)

\S4method{show}{Rcpp_MPCR}(object)
}
\arguments{
\item{x, object}{An MPCR objects.}

}
\description{
Prints the precision and type of the object, and print will print the meta data of the object without printing the values. Function x$PrintValues() should be used to print the values."
}
\details{
Prints metadata about the object and some values.
}
\value{
A string containing the metadata of the MPCR object.
}
\examples{
  \donttest{
    library(MPCR)
    x <- as.MPCR(1:16,4,4,"single")
    y <- as.MPCR(1:20,4,5,"double")
    x
    print(y)
    }
}
