
library(MPR)

paste("Create a Vector of 50 element with 32-Bit Precision")
x <- new(DataType,50,"float")


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
x$ToMatrix(5,10)
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
y <- new(DataType,10,"int")

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