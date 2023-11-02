##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

library(MPCR)

values <- c(1:24)
values
precision <- c("float", "double", "double", "float")


a <- matrix(values, 6, 4)
b <- matrix(c("double", "double", "double", "double"), 2, 2)

cat("Creating an MPCR TIle Object From a Matrix of Values and a Matrix of precisions")
cat("\n-------------------------------------------------------------------------------\n")
x <- new(MPCRTile, 6, 4, 3, 2, a, b)
paste("Printing Metadata: ")
paste("---------------------")
cat("Number of Rows : ")
x$Row
cat("\nNumber of Cols: ")
x$Col
cat("\nSize Of Matrix: ")
x$Size
cat("\nNumber of Rows in each Tile: ")
x$TileRow
cat("\nNumber of Cols in each Tile: ")
x$TileCol
cat("\nSize of Tile: ")
x$TileSize
cat("\n")
print(x)
paste("-----------------------------------------------------------")
paste("Print Tile 1,1")
x$PrintTile(1, 1)
paste("Print Tile 1,2")
x$PrintTile(1, 2)
paste("Print Tile 2,1")
x$PrintTile(2, 1)
paste("Print Tile 2,2")
x$PrintTile(2, 2)
paste("-----------------------------------------------------------")
paste("Change Precision of Tile 1,1 to Float")
x$ChangeTilePrecision(1, 1, "Float")
paste("Print Tile 1,1")
x$PrintTile(1, 1)
paste("-----------------------------------------------------------")
paste("Get Element 1,1")
x[1, 1]
paste("Set Element 1,1 to 1000")
x[1, 1] <- 1000
paste("Get Element 1,1")
x[1, 1]
x[1, 2]
paste("-----------------------------------------------------------")
paste("show Function")
x
paste("-----------------------------------------------------------")
paste("print Function")
print(x)

cat("----------------------------- Chol with no overwrite ------------------------------------\n")
a <- matrix(c(1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
              0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
              0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
              0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
              0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
              0.23, 0.07, 0.18, -0.08, -0.10, 0.36), 6, 6)
b <- c("float", "double", "float", "float",
       "double", "double", "float", "float",
       "double")

chol_mat <- new(MPCRTile, 6, 6, 2, 2, a, b)

chol_values <- chol(chol_mat, overwrite_input = FALSE, num_thread = 8)
print(chol_values)
cat("----------------------------- Chol with overwrite ------------------------------------\n")
chol(chol_mat)
print(chol_mat)


cat("----------------------------- Mat Mult ------------------------------------\n")

a <- matrix(c(3.12393, -1.16854, -0.304408, -2.15901,
              -1.16854, 1.86968, 1.04094, 1.35925,
              -0.304408, 1.04094, 4.43374, 1.21072,
              -2.15901, 1.35925, 1.21072, 5.57265), 4, 4)

zeros <- replicate(16, 0)

b <- c("float", "double", "float", "float")
c <- c("float", "float", "double", "float")


mat_a <- new(MPCRTile, 4, 4, 2, 2, a, b)
mat_b <- new(MPCRTile, 4, 4, 2, 2, a, c)
mat_c <- new(MPCRTile, 4, 4, 2, 2, zeros, c)

MPCRTile.gemm(a = mat_a, b = mat_b, c = mat_c, transpose_a = FALSE, transpose_b = FALSE, alpha = 1, beta = 1, num_thread = 8)
print(mat_c)

cat("----------------------------- DeepCopy ------------------------------------\n")
deep_copy_tile <- MPCRTile.copy(mat_c)
print(deep_copy_tile)


cat("----------------------------- Triangular Solve ------------------------------------\n")

mat_a <- new(MPCRTile, 4, 4, 2, 2, a, b)
mat_b <- new(MPCRTile, 4, 4, 2, 2, a, c)

MPCRTile.trsm(a = mat_a, b = mat_b, side = 'R', upper_triangle = TRUE, transpose = FALSE, alpha = 1)
print(mat_b)


cat("----------------------------- Sum and Product ------------------------------------\n")

a <- matrix(c(1:16), 4, 4)

mat_a <- new(MPCRTile, 4, 4, 2, 2, a, b)
print(mat_a)

main_diagonal <- mat_a$Diag()
cat("----------------------------- Main Diagonal of the matrix ------------------------------------\n")
main_diagonal$PrintValues()


sum <- mat_a$Sum()
cat("----------------------------- Sum of values from 1 to 16 ------------------------------------\n")
sum
prod <- mat_a$Prod()
cat("----------------------------- Product of values from 1 to 16 ------------------------------------\n")
prod

square_sum <- mat_a$SquareSum()
cat("----------------------------- Square Sum of values from 1 to 16 ------------------------------------\n")
square_sum

f_norm <- mat_a$Norm("f")
cat("----------------------------- Frobenius Norm of values from 1 to 16 ------------------------------------\n")
f_norm



