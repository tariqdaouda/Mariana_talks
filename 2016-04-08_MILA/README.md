#Talk @ the MILA lab at the University of Montreal on the 4th of April 2016

You can watch this talk on [youtube here](https://youtu.be/dGS_Qny1E9E):

This talk gives a detailed presentation of the implementation of various Mariana abstractions and shows how users can implement
their own. The live coding example shows how you can quickly make a neural network with one hidden layer and train it on mnist
using Mariana's default trainer. Using the trainer ensures that the model is saved in case of a crash, that the best models are
saved for test and/or validation sets. It also allows for the use of dataset mappers that take care of things such as 
oversampling, and recorders that generate and print reports during training.

To run the example you will need an internet connection to download the mnist dataset.
