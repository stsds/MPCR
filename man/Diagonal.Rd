\name{11-Diagonal}
\alias{diag}
\alias{diag,Rcpp_MPCR-method}
\title{diag}
\usage{
\S4method{diag}{Rcpp_MPCR}(x)
}
\arguments{
\item{x}{An MPCR matrix.}
}
\value{
An MPCR vector contains the main diagonal of the matrix.
}
\description{
Returns the diagonal of an MPCR matrix.
}
\examples{
    library(MPCR)
    x <- as.MPCR(1:16,4,4,"single")
    diag_vals <- diag(x)

}
