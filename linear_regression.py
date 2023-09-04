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

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/devshah/Documents/WorkSpace/CS Projects/Linear Regression/Student Study Hour V2.csv')


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m*x + b)) **2 
    total_error / float(len(points))
        