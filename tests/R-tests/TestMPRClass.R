library(MPR)

paste("Create a Vector of 50 element with 32-Bit Precision")
x <- new(MPR, 50, "float")


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
paste("Create a Vector of 10 element as INT")
y <- new(MPR, 10, "int")

paste("Element at Index 5")
y[[5]]
paste("Set Element at Index 5 to 10.55")
y[[5]] <- 10.55
paste("Element at Index 5 Should be 10 Because it's an int")
y[[5]]

paste("Vector Size")
y$Size

paste("Print all Elements in The Vector")
y$PrintValues()

paste("---------------------------------------------------------------")
paste("Get Min Value set First element with -1")
x[[0]] <- -1
min <- min(x)
min[[0]]

paste("Get Min Value Idx Should be 0")
min_idx <- which.min(x)
min_idx

paste("---------------------------------------------------------------")
paste("Get Max Value set First element with 10.55")
max <- max(x)
max[[0]]

paste("Get Min Value Idx Should be 5")
max_idx <- which.max(x)
max_idx

paste("---------------------------------------------------------------")
paste("Is Int Should be True")
is.sfloat(y)
paste("Is Int Should be False")
is.double(y)
is.float(y)

paste("---------------------------------------------------------------")
paste("Is NA Should be True")
x[[10]] <- 0
x[[11]] <- x[[10]] / 0
is.na(x, 11)
paste("Is NA Should be False")
is.na(x, 12)

paste("---------------------------------------------------------------")
paste("NA replace with 123")
na.exclude(x, 123)
x[[11]]
paste("NA omit size should be 49")
x[[11]] <- x[[10]] / 0
na.omit(x)
x$Size

paste("---------------------------------------------------------------")
paste("Replicate 1 2 3 (3 Times)")
temp_rep <- new(MPR, 3, "float")
temp_rep[[0]] <- 1
temp_rep[[1]] <- 2
temp_rep[[2]] <- 3

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
temp_bind <- new(MPR, 1, "float")
temp_bind[[0]] <- 22
replicated <- rep(temp_bind, 30)
replicated$ToMatrix(5, 6)

xx <- new(MPR, 30, "float")
xx$ToMatrix(5, 6)

paste("size should be 60")
cbind_temp <- cbind(xx, replicated)
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
cbind_temp <- rbind(xx, replicated)
cbind_temp$Size
cbind_temp$IsMatrix
paste("row should be 10")
cbind_temp$Row
paste("col should be 6")
cbind_temp$Col
cbind_temp$PrintValues()


paste("---------------------------------------------------------------")
paste("Sweep values should be 3 for all elements 1.5 * 2")

yy <- new(MPR, 10, "double")
temp_bind[[0]] <- 2
temp_sweep <- sweep(yy, temp_bind, 1, "+")
is.double(temp_sweep)

paste("---------------------------------------------------------------")
paste("Object size Vector of float 10 Element Float")
paste("Data size should be 40 byte + 13 metadata")
obj <- new(MPR, 10, "float")
size <- object.size(obj)
size

paste("Object size Vector of float 10 Element Double")
paste("Data size should be 80 byte + 13 metadata")
obj <- new(MPR, 10, "double")
size <- object.size(obj)
size

paste("---------------------------------------------------------------")
paste("Testing Default Print of MPR Object")
obj


paste("---------------------------------------------------------------")
paste("Testing Scale")

temp_scale <- new(MPR, 50, "float")
temp_scale$ToMatrix(5, 10)
for (val in 0:49) {
  temp_scale[[val]] <- val
}

temp_scale$PrintValues()
temp_center_scale <- new(MPR, 10, "double")
z <- scale(temp_scale, FALSE, temp_center_scale)
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Binary Operations")


x <- new(MPR, 50, "double")
y <- new(MPR, 50, "float")


for (val in 0:49) {
  x[[val]] <- val
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
z<- is.na(z)
z

paste("---------------------------------------------------------------")
paste("Testing Power Operations")
z <- x^y
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Diagonal on non Sqaure Matrix")
paste("Matrix")
x$ToMatrix(5,10)
x$PrintValues()
paste("Diagonal")
z<- diag(x)
z$PrintValues()


paste("---------------------------------------------------------------")
paste("Testing Replicate")
paste("Replicate (50, count=2) output =100")
paste("---------------------------------------------------------------")
x <- new(MPR,50,"float")
z <- rep(x,count=2)
z$PrintValues()

paste("Replicate (50, len=10) output =10")
paste("---------------------------------------------------------------")
z <- rep(x,len=10)
z$PrintValues()

