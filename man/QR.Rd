\name{33-QR decomposition}
\alias{qr}
\alias{qr,Rcpp_MPCR-method}
\alias{qr.Q,ANY-method}
\alias{qr.R,ANY-method}
\title{QR decomposition}
\usage{
\S4method{qr}{Rcpp_MPCR}(x, tol = 1e-07)

\S4method{qr.Q}{ANY}(qr, complete = FALSE, Dvec)

\S4method{qr.R}{ANY}(qr, complete = FALSE)

}
\arguments{
\item{x}{An MPCR matrix.}

\item{qr}{QR decomposition MPCR object.}

\item{tol}{The tolerance for determining numerical column rank.}


\item{complete}{Should the complete or truncated factor be returned?}

\item{Dvec}{Vector of diagonals to use when re-constructing Q (\bold{default is 1's}).}

}
\value{
 \item{qr}{Output of \code{qr()}.}
}
\description{
QR factorization and related functions.
}
\details{
The factorization is performed by the LAPACK routine \code{geqp3()}. This
should be similar to calling \code{qr()} on an ordinary R matrix with the
argument \code{LAPACK=TRUE}.
}
\examples{
\donttest{

library(MPCR)


qr_input <-as.MPCR( c(1, 2, 3, 2, 4, 6, 3, 3, 3),3,3,"single")
qr_out <- qr(qr_input)
qr_out
qr_out[["qr"]]$PrintValues()
qr_out[["qraux"]]$PrintValues()
qr_out[["pivot"]]$PrintValues()
qr_out[["rank"]]$PrintValues()

qr_q <- qr.Q(qr_out)
qr_q
}
}