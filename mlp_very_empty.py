import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

from Mariana.examples.useful import load_mnist

import numpy

if __name__ == "__main__" :
	
	#Define cost and learning scenario
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	#Define layers here
	inp = ML.Input(28*28, name = "InputLayer")

	hid = ML.Hidden(500,
		activations = MA.ReLU(),
		regularizations = [MR.L2(0.0001)],
		decorators = [MD.BinomialDropout(0.1)],
		name = "Hidden"
	)

	out = ML.SoftmaxClassifier(10,
		learningScenario = ls,
		costObject = cost,
		regularizations = [MR.L2(0.0001)],
		name = "OutputLayer"
	)

	#Define network here
	MLP = inp > hid > out

	#save html
	MLP.saveHTML("MLParis")

	train_set, validation_set, validation_set = load_mnist()

	trainScores = []
	miniBatchSize = 200
	for epoch in xrange(10000) :
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			inputs = train_set[0][i : i +miniBatchSize]
			targets = train_set[1][i : i +miniBatchSize]

			score = MLP.train("OutputLayer", InputLayer = inputs, targets = targets)
			trainScores.append(score)

		trainScore = numpy.mean(trainScores)
		
		valScore = MLP.test("OutputLayer", InputLayer = validation_set[0], targets = validation_set[1])
		print "---\nepoch: %s. train: %s, validation: %s" %(epoch, trainScore, valScore)


# ReLU, GradientDescent, NegativeLogLikelihood, BinomialDropout, SoftmaxClassifier, learningScenario, costObject, Input, InputLayer, Hidden, OutputLayerm saveHTML