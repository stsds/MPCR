\name{MMPRTile}
\alias{MMPRTile}
\title{MMPRTile S4 Class}

\description{
MMPRTile is a Mixed-Precision Tile Matrix data-structure with each tile having a specific precision.
\cr
Note: All Tiles must have the same size, and the total size must be divided equally on all tiles.
\cr
}

\section{Constructor}{
  \code{\link{new}} creates a new instance of Tile-Matrix \code{MMPRTile} class.
    \cr
  \code{new(MMPRTile,rows,cols,rows_per_tile,cols_per_tile,values,precisions)}
  \describe{
  \item{\code{rows}}{Number of rows in the Matrix}
  \item{\code{cols}}{Number of cols in the Matrix}
  \item{\code{rows_per_tile}}{Number of rows in each tile}
  \item{\code{cols_per_tile}}{Number of cols in each tile}
  \item{\code{values}}{R Matrix or vector containing all the values that should be in the Matrix}
  \item{\code{precisions}}{R Matrix or vector of string ,containing precision type of each tile}
}
}



\section{Accessors}{
  The following accessors can be used to get the values of the slots:

  \describe{
        \item{\code{Size}}{total number of elements inside the Matrix}
        \item{\code{Row}}{Number of rows}
        \item{\code{Col}}{Number of cols}
        \item{\code{TileRow}}{Number of rows in each tile}
        \item{\code{TileCol}}{Number of cols in each tile}
        \item{\code{TileSize}}{total number of elements in each tile}
  }
}

\section{Methods}{
  The following methods are available for objects of class \code{MMPRTile}:

  \subsection{PrintTile}{
    \code{PrintTile(tile_row_idx,tile_col_idx)}: Prints all the values stored inside a specific tile plus meta-data about the tile.
         \cr
     \describe{
        \item{\code{tile_row_idx}}{Row index of the Tile}
        \item{\code{tile_col_idx}}{Col index of the Tile}
      }
  }

  \subsection{ChangeTilePrecision}{
    \code{ChangeTilePrecision(tile_row_idx,tile_col_idx,precision)}: Change the precision of specific tile ,this function will need to copy all the values to cast them to the new precision
           \cr
         \describe{
            \item{\code{tile_row_idx}}{Row index of the Tile}
            \item{\code{tile_col_idx}}{Col index of the Tile}
            \item{\code{precision}}{required new precision as a string }
          }
    }

  \subsection{FillSquareTriangle}{
      \code{FillSquareTriangle(value,upper.tri,precision)}: Fills Upper or Lower Triangle
       with a given value and precision, New Tiles will be created,
       replacing the old tiles Note: the Input must be a square Matrix
          \cr
       \describe{
                   \item{\code{value}}{Value used during matrix filling}
                   \item{\code{upper.tri}}{Flag to indicate what triangle to fill. if TRUE, the upper triangle will be filled ,otherwise the lower triangle }
                   \item{\code{precision}}{the precision of the tiles created during matrix filling ,incase it's not a diagonal tile}
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
