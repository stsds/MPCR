
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

library("MPCR")

paste("Create a Vector of 50 element with 32-Bit Precision")
x <- new(MPCR, 50, "float","CPU")


paste("Element at Index 5")
x[[5]]
paste("Set Element at Index 5 to 10")
x[[5]] <- 10.55
paste("Element at Index 5")
x[[5]]

paste("Vector Size")
x$Size

paste("Is it a Matrix ?")
x$IsMatrix
paste("Row Count")
x$Row
paste("Col Count")
x$Col

paste("Represent as a Matrix with Row 5  Col 10")
x$ToMatrix(5, 10)
paste("Is it a Matrix ?")
x$IsMatrix
paste("Row Count")
x$Row
paste("Col Count")
x$Col
paste("Represent as a Vector")
x$ToVector()
paste("Is it a Matrix ?")
x$IsMatrix
paste("Row Count")
x$Row
paste("Col Count")
x$Col

paste("Print all Elements in The Vector")
x$PrintValues()

paste("---------------------------------------------------------------")
paste("Create a Vector of 10 element as HALF")
paste("If Compiler doesn't support 16-bit floating point ,32-bit will be created and a warning msg will appear")
y <- new(MPCR, 10, "half","CPU")

paste("Element at Index 5")
y[[5]]
paste("Set Element at Index 5 to 10.55")
y[[5]] <- 10.55
paste("Element at Index 5 Should be 10.55 if your compiler doesn;t support 16 bit")
y[[5]]

paste("Vector Size")
y$Size

paste("Print all Elements in The Vector")
y$PrintValues()

paste("---------------------------------------------------------------")
paste("Get Min Value set First element with -1")
x[[1]] <- -1
min <- min(x)
min[[1]]

paste("Get Min Value Idx Should be 0")
min_idx <- which.min(x)
min_idx

paste("---------------------------------------------------------------")
paste("Get Max Value set First element with 10.55")
max <- max(x)
max[[1]]

paste("Get Min Value Idx Should be 5")
max_idx <- which.max(x)
max_idx

paste("---------------------------------------------------------------")
paste("Is Int Should be True")
MPCR.is.half(y)
paste("Is Int Should be False")
is.double(y)
MPCR.is.float(y)

paste("---------------------------------------------------------------")
paste("Is NA Should be True")
x[[10]] <- 0
x[[11]] <- x[[10]] / 0
MPCR.is.na(x, 11)
paste("Is NA Should be False")
MPCR.is.na(x, 12)

paste("---------------------------------------------------------------")
paste("NA replace with 123")
MPCR.na.exclude(x, value=123)
x[[11]]
paste("NA omit size should be 49")
x[[11]] <- x[[10]] / 0
MPCR.na.omit(x)
x$Size

paste("---------------------------------------------------------------")
paste("Replicate 1 2 3 (3 Times)")
temp_rep <- new(MPCR, 3, "float","CPU")
temp_rep[[1]] <- 1
temp_rep[[2]] <- 2
temp_rep[[3]] <- 3

replicated <- rep(temp_rep, 9)
replicated$ToMatrix(3, 3)

paste("Size should be 9")
replicated$Size
replicated$PrintValues()

paste("---------------------------------------------------------------")
paste("Get Diagonal")
diag <- diag(replicated)
paste("size should be 3")
diag$Size
paste("values should be 1 , 2 , 3")
diag$PrintValues()

paste("---------------------------------------------------------------")
paste("CBind")
temp_bind <- new(MPCR, 1, "float","CPU")
temp_bind[[1]] <- 22
replicated <- rep(temp_bind, 30)
replicated$ToMatrix(5, 6)

xx <- new(MPCR, 30, "float","CPU")
xx$ToMatrix(5, 6)

paste("size should be 60")
cbind_temp <- MPCR.cbind(xx, replicated)
cbind_temp$Size
cbind_temp$IsMatrix
paste("row should be 5")
cbind_temp$Row
paste("col should be 12")
cbind_temp$Col
cbind_temp$PrintValues()

paste("---------------------------------------------------------------")
paste("RBind")

paste("size should be 60")
cbind_temp <- MPCR.rbind(xx, replicated)
cbind_temp$Size
cbind_temp$IsMatrix
paste("row should be 10")
cbind_temp$Row
paste("col should be 6")
cbind_temp$Col
cbind_temp$PrintValues()


paste("---------------------------------------------------------------")
paste("Sweep values should be 3 for all elements 1.5 * 2")

yy <- new(MPCR, 10, "double","CPU")
temp_bind[[1]] <- 2
temp_sweep <- sweep(x=yy, stat=temp_bind, margin=1, FUN="+")
MPCR.is.double(temp_sweep)

paste("---------------------------------------------------------------")
paste("Object size Vector of float 10 Element Float")
paste("Data size should be 40 byte + 13 metadata")
obj <- new(MPCR, 10, "float","CPU")
size <- MPCR.object.size(obj)
size

paste("Object size Vector of float 10 Element Double")
paste("Data size should be 80 byte + 13 metadata")
obj <- new(MPCR, 10, "double","CPU")
size <- MPCR.object.size(obj)
size

paste("---------------------------------------------------------------")
paste("Testing Default Print of MPCR Object")
obj


paste("---------------------------------------------------------------")
paste("Testing Scale")

temp_scale <- new(MPCR, 50, "float","CPU")
temp_scale$ToMatrix(5, 10)
for (val in 1:50) {
  temp_scale[[val]] <- val
}

temp_scale$PrintValues()
temp_center_scale <- new(MPCR, 10, "double","CPU")
z <- scale(x=temp_scale, center=FALSE, scale=temp_center_scale)
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Binary Operations")


x <- new(MPCR, 50, "double","CPU")
y <- new(MPCR, 30, "float","CPU")


for (val in 1:50) {
  x[[val]] <- val
}

for (val in 1:30) {
  y[[val]] <- val
}


paste("---------------------------------------------------------------")
paste("Testing == Operations")

z <- x == y
z

paste("---------------------------------------------------------------")
paste("Testing Plus Operations")
z <- x + y
z$PrintValues()

paste("---------------------------------------------------------------")
paste("Testing Minus Operations")
z <- x - y
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Multiply Operations")
z <- x * y
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Division Operations")
z <- x / y
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing IS Na Operations")
z <- MPCR.is.na(z)
z

paste("---------------------------------------------------------------")
paste("Testing Power Operations")
z <- x^y
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Diagonal on non Sqaure Matrix")
paste("Matrix")
x$ToMatrix(5, 10)
x$PrintValues()
paste("Diagonal")
z <- diag(x)
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Replicate")
paste("Replicate (50, count=2) output =100")
paste("---------------------------------------------------------------")
x <- new(MPCR, 50, "float","CPU")
z <- rep(x, count = 2)
z$PrintValues()

paste("Replicate (50, len=10) output =10")
paste("---------------------------------------------------------------")
z <- rep(x, len = 10)
z$PrintValues()

paste("---------------------------------------------- Linear Algebra --------------------------------------------------------")
values <- c(3.12393, -1.16854, -0.304408, -2.15901,
            -1.16854, 1.86968, 1.04094, 1.35925, -0.304408,
            1.04094, 4.43374, 1.21072, -2.15901, 1.35925, 1.21072, 5.57265)


eigen_temp <- c(1, -1, -1, 1)
x <- new(MPCR, 16, "float","CPU")
y <- new(MPCR, 16, "float","CPU")
z <- new(MPCR, 4, "float","CPU")


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

cat("----------------------- CrossProduct C=XY --------------------\n")
crossproduct <- crossprod(x, y)
crossproduct$PrintValues()
cat("----------------------- CrossProduct C=t(X)X --------------------\n")
crossproduct <- crossprod(x)
crossproduct$PrintValues()
cat("----------------------- %*% C=XY --------------------\n")
crossproduct <- x %*% y
crossproduct$PrintValues()
cat("----------------------- Eigen --------------------\n")
z$ToMatrix(2, 2)
z$PrintValues()
eigen_result <- eigen(z, FALSE)
eigen_result
valss <- eigen_result[[1]]
vecc <- eigen_result[[2]]

valss$PrintValues()
vecc$PrintValues()
cat("----------------------- QR Decomposition --------------------\n")
qr_vals <- c(1, 2, 3, 2, 4, 6, 3, 3, 3)
qr_input <- new(MPCR, 9, "float","CPU")
qr_input$ToMatrix(3, 3)


for (val in 1:9) {
  qr_input[[val]] <- qr_vals[[val]]
}
cat("----------------------- QR Decomposition Input--------------------\n")
qr_input
qr_input$PrintValues()

cat("----------------------- QR Decomposition Output--------------------\n")
qr_out <- qr(qr_input)
qr_out


qr_q <- qr.Q(qr_out)
qr_q
qr_out[[1]]$PrintValues()
qr_out[[2]]$PrintValues()
qr_out[[3]]$PrintValues()
qr_out[[4]]$PrintValues()

cat("----------------------- SVD --------------------\n")
svd_vals <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1)

svd_input <- new(MPCR, 9 * 4, "float","CPU")
svd_input$ToMatrix(9, 4)

for (val in 1:36) {
  svd_input[[val]] <- svd_vals[[val]]
}
cat("----------------------- SVD Input--------------------\n")
svd_input$PrintValues()

svd_output <- svd(svd_input)
cat("----------------------- SVD Output--------------------\n")
svd_output
svd_output$d$PrintValues()
svd_output[[2]]$PrintValues()
svd_output[[3]]$PrintValues()

cat("------------------------------- RCond ------------------------------------------\n")
#
rcond_out <- rcond(svd_input, "O", FALSE)
rcond_out


cat("--------------------------------------------------------------------------\n")
cat("------------------  as.MPCR Function ------------------------\n")
convertedMPCR <- as.MPCR(1:24, precision = "float",placement="CPU")
convertedMPCR$PrintValues()
cat("--------------------------------------------------------------------------\n")
convertedMPCR <- as.MPCR(1:24, nrow = 4, ncol = 6, precision = "float",placement="CPU")
convertedMPCR$PrintValues()
cat("-------------- Test Print --------------------------\n")
print(convertedMPCR)


sum <- convertedMPCR$Sum()
cat("----------------------------- Sum of values from 1 to 24 ------------------------------------\n")
sum
prod <- convertedMPCR$Prod()
cat("----------------------------- Product of values from 1 to 24 ------------------------------------\n")
prod

square_sum <- convertedMPCR$SquareSum()
cat("----------------------------- Square Sum of values from 1 to 24 ------------------------------------\n")
square_sum



