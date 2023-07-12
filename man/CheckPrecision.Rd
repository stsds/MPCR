\name{Check Precision}
\alias{Check Precision}
\alias{MMPR.is.single.mode,Rcpp_MMPR-method}
\alias{MMPR.is.double,Rcpp_MMPR-method}
\alias{MMPR.is.half,Rcpp_MMPR-method}
\alias{MMPR.is.float,Rcpp_MMPR-method}

\title{Metadata functions}
\description{
    Checks the precision of a given MMPR object.
}
\usage{

\S4method{MMPR.is.single}{Rcpp_MMPR}(x)
\S4method{MMPR.is.half}{Rcpp_MMPR}(x)
\S4method{MMPR.is.double}{Rcpp_MMPR}(x)
\S4method{MMPR.is.float}{Rcpp_MMPR}(x)

}
\arguments{
  \item{x}{
    An MMPR object.
  }
}
\value{
 Boolean indicates the precision of the object according to the used function.
}
\examples{
  \dontrun{
  library(MMPR)

  x <- as.MMPR(1:20,precision="double")
  MMPR.is.double(x) #TRUE
  MMPR.is.single(x) #FALSE
  }
}