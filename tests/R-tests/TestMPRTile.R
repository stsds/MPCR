library(MPR)

values <- c(1:24)
values
precision <- c("float","double","double","float")


a <- matrix(values, 6,4)
b <- matrix(c("double","double","double","double"), 2,2)

cat("Creating an MPR TIle Object From a Matrix of Values and a Matrix of precisions")
cat("\n-------------------------------------------------------------------------------\n")
x <- new(MPRTile,6,4,3,2,a,b)
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
paste("-----------------------------------------------------------")
paste("Print Tile 0,0")
x$PrintTile(0,0)
paste("Print Tile 0,1")
x$PrintTile(0,1)
paste("Print Tile 1,0")
x$PrintTile(1,0)
paste("Print Tile 1,1")
x$PrintTile(1,1)
paste("-----------------------------------------------------------")
paste("Change Precision of Tile 0,0 to Float")
x$ChangeTilePrecision(0,0,"Float")
paste("Print Tile 0,0")
x$PrintTile(0,0)
paste("-----------------------------------------------------------")
paste("Get Element 1,1")
x[1,1]
paste("Set Element 1,1 to 1000")
x[1,1] <- 1000
paste("Get Element 1,1")
x[1,1]
x[1,2]
paste("-----------------------------------------------------------")
paste("show Function")
x
paste("-----------------------------------------------------------")
paste("print Function")
print(x)
