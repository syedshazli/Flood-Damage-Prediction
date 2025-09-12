# import numpy as np
# import torch
# import torch.nn as nn # Pytorch neural network modules
# import torch.nn.functional as F # activation functions 
# from torch.optim import SGD # Stochastic gradient descent


# # Plotting
# import seaborn as sns

# # Inherits from Pytorch class module
# class BasicNN(nn.Module):
#     @torch.no_grad()
#     def __init__(self):
        

#         # Inherits from parent class
#         super().__init__()

#         # create neural network parameters (weights & biases)
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad = False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad = False)

#         # Requires gradient defaults to true
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad = False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad = False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad = False)

#         # the final bias before it enters relu for prediction
#         self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

#     # Make forward pass
#     def forward(self,input):
#         # connect input to activation function
#         # Equal to w*x + b
#         inputToReluZero = input * self.w00 + self.b00
#         # Pass in this to relu
#         reluZeroOutput = F.relu(inputToReluZero) 
#         scaledReluZeroOutput = reluZeroOutput * self.w01

#         inputToReluOne = input * self.w10 + self.b10
#         reluOneOutput = F.relu(inputToReluOne)
#         scaledReluOneOutput = reluOneOutput * self.w11

#         # add top and bottom scaled values of NN to input as final relu, along w/bias
#         inputForFinalRelu = scaledReluOneOutput + scaledReluZeroOutput + self.final_bias

#         output = F.relu(inputForFinalRelu)

#         return output



# input = torch.linspace(start = 0, end = 1, steps = 11)

# print(input)

# model = BasicNN()
# # automatically calls forward function when init the model with data
# outputVals = model(input)
# print(outputVals)


# sns.set(style="whitegrid")
# sns.lineplot(x=input, y=outputVals, color='green', linewidth=2.5)

# plt.ylabel('Effectivness')
# plt.xlabel('Dosage')

import matplotlib.pyplot as plt

# Data for the plot
x_values = [1, 2, 3, 4, 5]
y_values = [2, 4, 1, 5, 3]

# Create the plot
plt.plot(x_values, y_values)

# Add labels and a title for clarity
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Simple Line Plot")

# Display the plot
plt.show()