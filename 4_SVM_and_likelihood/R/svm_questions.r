
# Consider two classes of points which are well separated

# Reading in dataset (you will have to change the working directory path)
setwd("~/Documents/PhD/teaching/Machine_learning_2020/R/")
df <- read.csv(file = 'make_blobs_1.csv')
df2 <- read.csv(file = 'make_circles_1.csv')
df3 <- read.csv(file = 'make_circles_2.csv')

plot(df$X_x, df$X_y, col = df$labels+3)

# Task 1: Attempt to use linear regression to separate this data using linear regression.
# Note there are several possibilities which separate the data? 
# What happens to the classification of point [0.6, 2.1] (or similar)?



# With SVM rather than simply drawing a zero-width line between the 
# classes, we draw a margin of some width around each line, up to the nearest point. 
# For example for these lines:

# xfit <- seq(-1, 3.5, length.out=50)
# plot(df$X_x, df$X_y, col = df$labels+3)


# for (i in list(c(1, 0.65, 0.33), c(0.5, 1.6, 0.55), c(-0.2, 2.9, 0.2))){
#   yfit <- i[1]*xfit + i[2]
#   lines(xfit, yfit, type = "l")
#   lines(xfit, yfit+i[3], type="l", lty = 3, col = 'orange')
#   lines(xfit, yfit-i[3], type="l", lty = 3, col = 'orange')
# }

# Task 2: Draw the margin around the lines you chose in Task 1.




# For SVM the line that maximises the margin is the optimal model

# Task 3: Use the e1071 package to build a support vector classifier using a linear kernel
# Plot the decision fuction on the data

install.packages("e1071")
library("e1071")



# Task 4: Change the number of points in the dataset using X = X[1:N] and df$labels = df$labels[1:N]
# and build the classifier again using a linear kernel
# Plot the decision function. Do you see any differences?



## So far we have considered linear boundaries but this is not always the case

## Consider the new dataset df2

plot(df2$X_x, df2$X_y, col = df2$labels+3)

#Task 5: Build a classifier using a linear kernel and plot the decision making function


# These results should look wrong so we will try something else

# Consider projecting our data into a 3D plane

r=c()
for (i in 1:length(df2$X_x)) { 
  r[i]=exp(-(df2$X_x[i] ** 2+df2$X_y[i]** 2))
}

install.packages("scatterplot3d") # Install
library("scatterplot3d") # load

scatterplot3d(x = df2$X_x, y=df2$X_y, z=r, color = df2$labels+3)

# Looking at the data it is now clear to see that we could draw a linear plane through
# it in the 3D space and classify the data. We can then project back to the 2D
# space. This is what the 'rbf' kernel does.

#Task 6: Try building a classifier using the 'rbf' kernel


# Task 7: Go back to your original dataset (ie. make blobs) and try using different kernels 
# to build the classifier and plot the results
# Compare the differences between the models



## So far we have looked at clearly delineated data. Consider the following dataset
## where the margins are less clear

plot(df3$X_x, df3$X_y, col = df3$labels+3)

## SVM has a tuning parameter C which softerns the margins. For very large C, 
## the margin is hard, and points cannot lie in it. For smaller $C$, the margin 
# is softer, and can grow to encompass some points.

# Task 8: Try experimenting with different values of C and see what different
# results you get



# Task 9: Use GridSearchCV from superml to find the optimum parameters for C. 

