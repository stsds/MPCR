
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MMPR is an R package provided by the STSDS group at KAUST
##########################################################################

library(MMPR)

a <- matrix(c(1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
              0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
              0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
              0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
              0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
              0.23, 0.07, 0.18, -0.08, -0.10, 0.36), 6, 6)
b <- c("double", "double", "double", "double",
       "double", "double", "double", "double",
       "double")


chol_mat <- new(MMPRTile, 6, 6, 2, 2, a, b)
num_rows_tile <- chol_mat$Row / chol_mat$TileRow


for (k in 1:num_rows_tile) {
  tile_chol <- MMPRTile.GetTile(matrix = chol_mat, row = k, col = k)
  potrf_output <- chol(x = tile_chol, upper_triangle = FALSE) # matrix upper row col
  MMPRTile.UpdateTile(matrix = chol_mat, tile = potrf_output, row = k, col = k)

  if (k < num_rows_tile) {
    for (i in (k + 1):(num_rows_tile)) {
      tile_one <- MMPRTile.GetTile(matrix = chol_mat, row = k, col = k)
      tile_two <- MMPRTile.GetTile(matrix = chol_mat, row = i, col = k)
      trsm_output <- MMPR.trsm(a = tile_one, b = tile_two, upper_triangle = FALSE, transpose = TRUE, side = 'R', alpha = 1)
      MMPRTile.UpdateTile(matrix = chol_mat, tile = trsm_output, row = i, col = k)
    }


    for (j in (k + 1):(num_rows_tile)) {
      tile_one <- MMPRTile.GetTile(matrix = chol_mat, row = j, col = k)
      tile_two <- MMPRTile.GetTile(matrix = chol_mat, row = j, col = j)
      MMPR.gemm(a = tile_one, c = tile_two, alpha = -1, beta = 1)
      MMPRTile.UpdateTile(matrix = chol_mat, tile = tile_two, row = j, col = j)

      if (j < num_rows_tile) {
        for (i in (j + 1):(num_rows_tile)) {
          tile_one <- MMPRTile.GetTile(matrix = chol_mat, row = i, col = k)
          tile_two <- MMPRTile.GetTile(matrix = chol_mat, row = j, col = k)
          tile_three <- MMPRTile.GetTile(matrix = chol_mat, row = i, col = j)
          MMPR.gemm(a = tile_one, b = tile_two, c = tile_three, transpose_b = TRUE, alpha = -1, beta = 1)
          MMPRTile.UpdateTile(matrix = chol_mat, tile = tile_three, row = i, col = j)
        }
      }
    }
  }

}
chol_mat$FillSquareTriangle(0, TRUE, "double")
print(chol_mat)

chol_mat_val <- new(MMPRTile, 6, 6, 2, 2, a, b)
chol(chol_mat_val)
print(chol_mat_val)