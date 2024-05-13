
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################


library("MPCR")

paste("---------------------------------------------- Linear Algebra --------------------------------------------------------")
values <- c(3.12393, -1.16854, -0.304408, -2.15901,
            -1.16854, 1.86968, 1.04094, 1.35925, -0.304408,
            1.04094, 4.43374, 1.21072, -2.15901, 1.35925, 1.21072, 5.57265)


eigen_temp <- c(1, -1, -1, 1)
x <- new(MPCR, 16, "float","GPU")
y <- new(MPCR, 16, "float","GPU")
z <- new(MPCR, 4, "float","GPU")


for (val in 1:16) {
  x[[val]] <- values[[val]]
  y[[val]] <- values[[val]]

}

for (val in 1:4) {
  z[[val]] <- eigen_temp[[val]]
}
x$ToMatrix(4, 4)
y$ToMatrix(4, 4)
paste("X and Y values")
x$PrintValues()
y$PrintValues()

MPCR.SetOperationPLacement("GPU")
cat("----------------------- CrossProduct C=XY --------------------\n")
crossproduct <- crossprod(x, y)
crossproduct$PrintValues()