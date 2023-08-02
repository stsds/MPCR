\name{MMPRTile}
\alias{MMPRTile}
\title{MMPRTile S4 Class}

\description{
MMPRTile is a data structure for tile matrices with mixed precision, where each tile possesses a specific precision level.
}

\section{Constructor}{
  \code{\link{new}} creates a new instance of Tile-Matrix \code{MMPRTile} class.
  \cr
  \code{new(MMPRTile,rows,cols,rows_per_tile,cols_per_tile,values,precisions)}
  \describe{
  \item{\code{rows}}{Number of rows in the matrix.}
  \item{\code{cols}}{Number of cols in the matrix.}
  \item{\code{rows_per_tile}}{Number of rows in each tile.}
  \item{\code{cols_per_tile}}{Number of cols in each tile.}
  \item{\code{values}}{R matrix or vector containing all the values that should be in the matrix.}
  \item{\code{precisions}}{R matrix or vector of strings, containing precision type of each tile.}
}
}



\section{Accessors}{
  The following accessors can be used to get the values of the slots:

  \describe{
        \item{\code{Size}}{Total number of elements inside the Matrix.}
        \item{\code{Row}}{Number of rows.}
        \item{\code{Col}}{Number of cols.}
        \item{\code{TileRow}}{Number of rows in each tile.}
        \item{\code{TileCol}}{Number of cols in each tile.}
        \item{\code{TileSize}}{Total number of elements in each tile.}
  }
}

\section{Methods}{
  The following methods are available for objects of class \code{MMPRTile}:

  \subsection{PrintTile}{
  \cr
    \code{PrintTile(tile_row_idx,tile_col_idx)}: Prints all the values stored inside a specific tile plus meta-data about the tile.
         \cr
     \describe{
        \item{\code{tile_row_idx}}{Row index of the tile.}
        \item{\code{tile_col_idx}}{Col index of the tile.}
      }
  }

  \subsection{ChangeTilePrecision}{
  \cr
    \code{ChangeTilePrecision(tile_row_idx,tile_col_idx,precision)}: Change the precision of specific tile, this function will need to copy all the values to cast them to the new precision.
           \cr
         \describe{
            \item{\code{tile_row_idx}}{Row index of the tile.}
            \item{\code{tile_col_idx}}{Col index of the tile.}
            \item{\code{precision}}{Required new precision as a string.}
          }
    }

  \subsection{FillSquareTriangle}{
  \cr
      \code{FillSquareTriangle(value,upper.tri,precision)}: Fills upper or lower triangle
       with a given value and precision, new tiles will be created,
       replacing the old tiles. \bold{Note:} The input must be a square matrix
          \cr
       \describe{
                   \item{\code{value}}{ A value used during matrix filling.}
                   \item{\code{upper.tri}}{A flag to indicate what triangle to fill. if TRUE, the upper triangle will be filled, otherwise the lower triangle. }
                   \item{\code{precision}}{The precision of the tiles created during matrix filling, in case it's not a diagonal tile.}
                 }
  }

   \subsection{Sum}{
        \code{Sum()}: Get the sum of all elements in all tiles in MMPRTile Matrix.
    }

     \subsection{Prod}{
          \code{Prod()}: Get the product of all elements in all tiles in MMPRTile Matrix.
      }
}

\examples{
  \dontrun{
    # Example usage of the class and its methods
   a <- matrix(1:36, 6, 6)
   b <- c("double", "double", "single", "double",
          "half", "double", "half", "double",
          "single")

   tile_mat <- new(MMPRTile, 6, 6, 2, 2, a, b)
   tile_mat
   sum <- tile_mat$Sum()
   prod <- tile_mat$Prod()
   tile_mat$PrintTile(1,1)
   tile_mat$ChangeTilePrecision(1,1,"single")

   n_rows <- tile_mat$Row
   n_cols <- tile_mat$Col
   total_size <- tile_mat$Size
   rows_per_tile <- tile_mat$TileRow
   cols_per_tile <- tile_mat$TileCol
  }
}

\keyword{S4 class}
