library(MPR)

x <- new(MPR, 50, "double")
y <- new(MPR, 50, "float")

# class(x)

for (val in 0:49) {
  x[[val]] <- val
  y[[val]] <- val
}

MPR
ls("package:MPR")

z <- x + y


# z <-Plus(x,y)
# z$PrintValues()
