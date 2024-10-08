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


x <- as.MPCR(values, nrow = 4, ncol = 4, precision = "single", placement = "CPU")
y <- as.MPCR(values, nrow = 4, ncol = 4, precision = "single", placement = "CPU")


paste("X and Y values")
x$PrintValues()
y$PrintValues()

# Perform all the upcoming operation on GPU, if supported.
MPCR.SetOperationPlacement("GPU")
cat("----------------------- CrossProduct C=XY --------------------\n")
# Data will be transfered automatically to GPU to be able to perform the operation
# on GPU
crossproduct <- crossprod(x, y)
crossproduct$PrintValues()