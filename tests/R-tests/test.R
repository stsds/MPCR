library("MPCR")
paste("*****************************************************************")
paste("Create a Vector of 50 element with 32-Bit Precision")

s1 <- as.MPCR(1:20,nrow=2,ncol=10,"single")
s2 <- as.MPCR(21:40,nrow=2,ncol=10,"double")
x <- s1 + s2
typeof(x)
x$PrintValues()



