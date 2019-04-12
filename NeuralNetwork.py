#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:17:55 2019

@author: user
"""
import numpy as np
import pandas as pd
import math
import json

def sigmoid(x):
    return (1/(1 + math.exp(-x)))


def derivativeOfSigmoid(x):
##    return sigmoid(x)*(1-sigmoid(x))
    return x * (1 - x)
    
    
class NeuralNetwork:
    
    
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons):
        self.inputNeurons = inputNeurons
        self.hiddenNeurons = hiddenNeurons
        self.outputNeurons = outputNeurons
        self.w_ih = np.random.rand(hiddenNeurons, inputNeurons)
        self.w_ho = np.random.rand(outputNeurons, hiddenNeurons)
        self.bias_h = np.random.rand(hiddenNeurons, 1)
        self.bias_o = np.random.rand(outputNeurons,1)
        self.lr = 0.3
    
        
       
    
    def feedForward(self, inputs):
        inp = np.asmatrix(inputs)
    #    print("w_ih", self.w_ih)
   #     print("w_ho", self.w_ho)
        
        hidden = np.dot(self.w_ih, np.transpose(inp))
    #    print("hidden", hidden) 
        np.add(hidden, self.bias_h)
        
        ##activation function to be multiplied
        vectorize = np.vectorize(sigmoid)
        hidden = vectorize(hidden)
  #      print("hidden after vectorization", hidden)
    ##    hidden = hidden(sigmoid)
        
        
        output = np.dot(self.w_ho, hidden)
        np.add(output, self.bias_o)
        
        output = vectorize(output)
        return np.asarray(output)
    
    def train(self, inputs, targets):
        
        inp = np.asmatrix(inputs)
     #   print("inp as array", inp)
     #   print("np.transpose(inp)",np.transpose(inp))
        hidden = np.dot(self.w_ih, np.transpose(inp))
        np.add(hidden, self.bias_h)
        
        ##activation function to be multiplied
        vectorizedSigmoid = np.vectorize(sigmoid)
        vectorizedSigmoidDerivative = np.vectorize(derivativeOfSigmoid)
        hidden = vectorizedSigmoid(hidden)
    ##    hidden = hidden(sigmoid)
        
        
        output = np.dot(self.w_ho, hidden)
        np.add(output, self.bias_o)
        
        output = vectorizedSigmoid(output)
        
        
        outputForGivenInput = self.feedForward(inputs)
        outputForGivenInput = np.asarray(outputForGivenInput)
        targets = np.asarray(targets)
        ##error = targets - outputs
        output_error = np.subtract(targets, outputForGivenInput)
        print("output-error", output_error)
        
        # calculate gradients
        gradient_ho = vectorizedSigmoidDerivative(output)
        gradient_ho = np.multiply(gradient_ho, output_error)
        gradient_ho = np.multiply(gradient_ho, self.lr)
        gradient_ho = np.transpose(gradient_ho)
        
        hidden_transpose = np.transpose(hidden)
        # calculate weight_ho deltas
        weights_ho_deltas = np.dot(gradient_ho, hidden_transpose)
        
        self.w_ho = np.add(self.w_ho, weights_ho_deltas)
        
        ## adjust biases
        self.bias_o = np.add(self.bias_o, gradient_ho)
        
        hidden_errors = np.multiply(np.transpose(self.w_ho), output_error)
        # calculate gradients
        gradient_ih = vectorizedSigmoidDerivative(hidden)
        gradient_ih = np.multiply(gradient_ih, hidden_errors)
        gradient_ih = np.multiply(gradient_ih, self.lr)
        gradient_ih = np.transpose(gradient_ih)
        inp_transpose = np.transpose(inp)
        # calculate weight-iH deltas
        weights_ih_deltas = np.dot(gradient_ih, inp_transpose)
        self.w_ih = np.add(self.w_ih, weights_ih_deltas)
        
        self.bias_h = np.add(self.bias_h, gradient_ih)

        
    
    
training_data = [
        ([0,0],
         [0]
         ),
        ([0,1],
         [1]
         ),
        ([1,0],
         [1]
         ),
        ([1,1],
         [0]
         )
        ]
 

nn = NeuralNetwork(2,2,1)
print("hello.. in NN setup")
inputs = [1,0]
targets = [1]
inp = [1,0]


#output = nn.feedForward(inp)
#result = nn.train(inp, targets)

for x in range(1000):
    for data in training_data:
        nn.train(data[0], data[1])
        
guess = nn.feedForward([0,0])


print("printin guess", guess)

