\name{diag}
\alias{diag}
\alias{diag,Rcpp_MMPR-method}
\title{diag}
\usage{
\S4method{diag}{Rcpp_MMPR}(x)
}
\arguments{
\item{x}{A MMPR Matrix}
}
\value{
A MMPR vector containing the main diagonal of the matrix
}
\description{
Methods for getting the diagonal of a MMPR matrix
}
\examples{
  \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    diag_vals <- diag(x)
    }

}
