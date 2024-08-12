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


x <- as.MPCR(values, nrow = 4, ncol = 4, precision = "float", placement = "CPU")
y <- as.MPCR(values, nrow = 4, ncol = 4, precision = "float", placement = "CPU")


paste("X and Y values")
x$PrintValues()
y$PrintValues()

MPCR.SetOperationPlacement("GPU")
cat("----------------------- CrossProduct C=XY --------------------\n")
crossproduct <- crossprod(x, y)
crossproduct$PrintValues()