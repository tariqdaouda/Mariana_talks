#Talk @ the MILA lab at the University of Montreal on the 4th of April 2016

This talk gives a detailed output of the implementation of various Mariana abstractions and show how users can implement
their own. The live codind example shows how you can quickly make a neural network with one hidden layer and train it on mnist
using Mariana's default trainer. Using the trainer ensures that the model is saved in case of a crash, that the best models are
saved for test and/or validation sets. It also allows for the use of dataset mappers that take care of things such as 
oversampling, and recorders that generate and print reports during training.
