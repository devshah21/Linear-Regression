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
    
def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        m_gradient += -(2/n) * x * (y -(m_now * x + b_now))
        b_gradient += -(2/n) * (y -(m_now * x + b_now))
        
    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    return m, b

m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)

print(m,b)

plt.scatter(data.studytime, data.score, color='black')
plt.plot(list(range(1, 10)), [m*x + b for x in range(1,10)], color='red')

        