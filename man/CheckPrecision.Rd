\name{25-Check precision}
\alias{Check Precision}
\alias{MPCR.is.single,Rcpp_MPCR-method}
\alias{MPCR.is.double,Rcpp_MPCR-method}
\alias{MPCR.is.half,Rcpp_MPCR-method}
\alias{MPCR.is.float,Rcpp_MPCR-method}

\title{Metadata functions}
\description{
    Checks the precision of a given MPCR object.
}
\usage{

\S4method{MPCR.is.single}{Rcpp_MPCR}(x)
\S4method{MPCR.is.half}{Rcpp_MPCR}(x)
\S4method{MPCR.is.double}{Rcpp_MPCR}(x)
\S4method{MPCR.is.float}{Rcpp_MPCR}(x)

}
\arguments{
  \item{x}{
    An MPCR object.
  }
}
\value{
 Boolean indicates the precision of the object according to the used function.
}
\examples{
  library(MPCR)
  x <- as.MPCR(1:20,precision="double")
  MPCR.is.double(x) #TRUE
  MPCR.is.single(x) #FALSE
}