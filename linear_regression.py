# linear regression is all about minimizing the mean-squared error
# i.e the distance from the actual value of the function vs. the value on the graph
# error = 1/n * (summation (y_i - (m*x_i +b)^2)) 
# y_i is the actual y value (the graph) and subtract it from the function's y value
# we divide by n to get the mean squared error
# we want to minimize the error function in line 3
# we want to find m and b s.t we minimize error
# we can do that by taking the partial derivative w.r.t to m and b
# that gives us the direction of steepest ascent 
# refer to the image in the repo for the math
