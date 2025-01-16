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

# Perform all the upcoming operation on GPU, if supported.
# default stream is used
# Set the operation placement of the default stream to GPU
MPCR.SetOperationPlacement("default", "GPU")

# Create a new ASYNC stream named "gemm" on the GPU
MPCR.CreateStream("gemm", "GPU" ,"ASYNC")
MPCR.SetRunMode("gemm", "async")

# Create a new SYNC stream named "chol" on the GPU
MPCR.CreateStream("chol", "GPU" ,"SYNC")
MPCR.SetRunMode("chol", "sync")

crossproduct_result <- crossprod(x, y)

# Set the "gemm" stream as the operation stream
MPCR.SetOperationStream("gemm")

crossproduct <- crossprod(x, y)
# Synchronize the "gemm" stream
MPCR.SyncStream("gemm")

# Set the "chol" stream as the operation stream
MPCR.SetOperationStream("chol")
chol_result <- chol(x)

# Delete the "gemm" stream
MPCR.DeleteStream("gemm")
# Delete the "chol" stream
MPCR.DeleteStream("chol")

# Set the "default" stream as the operation stream
MPCR.SetOperationStream("default")
