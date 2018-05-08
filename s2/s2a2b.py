#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:43:24 2018

@author: gopsmi
"""

import numpy as np

# Store the input and output in X and y, respectively

X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
print(np.shape(X))
print(X)

y = np.array([[1], [1], [0]])

# Define weights (wh_1) for the 1st hidden layer that connects with the
# input layer and define the bias (bh_1) for the hidden layer. Both the 
# weights and biases for the hidden layer were defined using random number
# generation method using numpy operation -- np.random.random()

# Before generating random numbers using np.random.random(), use the function
# np.random.seed() to ensure repeatability of the number generation.
# This will help us cross-check results with other systems, and is convenient
# to also manually calculate and cross-check with the results from scripts.

np.random.seed(1234)
wh_1 = np.round(np.random.random((4, 3)), 2)
print(wh_1)

# bias values are sorted in the descending order (just to give higher values
# initially to the first two X values -- input, and attach low significance to the 
# last input value -- to reflect the output y values [1 1 0])
# though this is not needed -- just a guess, the back propagation will adjust
# if its a wrong guess.

np.random.seed(1234)
bh_1 = -np.sort(-np.round(np.random.random((1, 3)), 2), axis=-1)
print(bh_1)
np.random.seed(1234)
wout = np.random.random((3, 1))
print(wout)
np.random.seed(1234)
bout = np.random.random((1,1))
print(bout)

# To calculate hidden layer input which is a biased weighted sum of inputs
hidden_layer_input = np.dot(X, wh_1) + bh_1
print(hidden_layer_input)

# To evaluate activation function for the computed input
# First we will define the sigmoid function which is used in converting the 
# hidden layer input to intermediate output which is then fed into the next layer
# as input. The sigmoid is one among many activation functions used in the 
# Neural Network algorithms. The output from sigmoid is in the range of [0, 1].

def sigmoid(x):
    return 1/(1+np.exp(-x))

hiddenlayer_activations = np.round(sigmoid(hidden_layer_input), 2)
print(hiddenlayer_activations)

# This hidden layer activation goes as input to the next layer -- which is our
# final output layer in this case (we can have multiple intermediate hidden layers
# most often). Since this is our simple case study on how to implement backward
# propagation we have limited oursellves to deal with 3 layers i.e. one input layer,
# one hidden layer and one output layer.

outer_layer_input =np.dot(hiddenlayer_activations, wout) + bout
print(outer_layer_input)

# Finally sigmoid of the above input is our final output for the first cycle,
# and we calculate error based on the known actual output and the output from the
# output layer, which is then used to backpropagate to fine tune the weights
# and biases so as to reduce the error in the final output.

output = np.round(sigmoid(outer_layer_input), 2)
print(output)

error_out = y - output

# Compute the slope at output and hidden layer.
# slope_output_layer = derivatives_sigmoid(output) = output*(1-output)
# slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) = 
# hiddenlayer_activations(1-hiddenlayer_activations)

# also define the learning rate (lrng_rate), the scale for incrementing error 
# values while updating the output at the previous layer

lrng_rate = 0.1

slope_output_layer = np.round(output*(1-output), 2)
print(slope_output_layer)

slope_hidden_layer = np.round(hiddenlayer_activations*(1-hiddenlayer_activations), 2)
print(slope_hidden_layer)

# Now compute the change or increment of the error value based on the slope
# computed above. This is called the error responsibility of the node from the 
# input layer (or intermediate hidden layer that serves as input to the next layer).

# Error responsibility = (1-ouput)*output*(actual output - computed output) for 
# the output layer.

# Error responsbility (for nodes in hidden/intermediate layers) is given by,
# error resp = output*(1-output)*Summation of Wjk*deltaj, where the 
# summation of Wjk*deltaj represents the error responsbilities for nodes downstream
# from the particular hidden layer node.


d_output = np.round(slope_output_layer*error_out, 4)

# To calculate error at hidden layer, the partial derivatives of error from the
# output is computed at several points -- like partial derivative (here after, 
# p.d) of error w.r.t output layer output, p.d of output layer output to 
# output layer input (p.d of activation function), p.d of output layer input
# to hidden layer output, p.d of hidden layer output to its input (p.d of 
# activation function), p.d of hidden layer input to input layer weighted coefficients.

error_hd_layer = np.dot(d_output, wout.T)

print(error_hd_layer)

# Change or increment in the weights of hidden layer is computed.

d_hiddenlayer = np.round(slope_hidden_layer*error_hd_layer, 3)

# Finally the change or increments in the respective weights/biases are added to
# their original or last updated weights/biases scaled by the learning rate.

last_wout = wout
last_wh_1 = wh_1
last_bh_1 = bh_1
last_bout = bout

wout = wout + np.dot(hiddenlayer_activations.T, d_output)*lrng_rate
wh_1 = wh_1 + np.dot(X.T, d_hiddenlayer)*lrng_rate

bh_1 = bh_1 + np.sum(d_hiddenlayer, axis=0)*lrng_rate
bout = bout + np.sum(d_output, axis=0)*lrng_rate

print("Original or lastly updated hidden layer weights/biases are \n", last_wh_1,"\n and \n", last_bh_1)
print("Original or lastly updated output layer weights/biases are \n", last_wout,"\n and \n", last_bout)



































































