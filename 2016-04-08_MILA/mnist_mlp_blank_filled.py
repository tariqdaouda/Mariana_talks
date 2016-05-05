import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.training.trainers as MT
import Mariana.training.datasetmaps as MDM
import Mariana.training.stopcriteria as MSTOP

from Mariana.examples.useful import load_mnist

if __name__ == "__main__":

	# Define the network
	ls = MS.MomentumGradientDescent(lr = 0.01, momentum=0.9)
	cost = MC.NegativeLogLikelihood()

	inputLayer = ML.Input(28*28, name = "InputLayer")
	
	h1 = ML.Hidden(100,
		activation=MA.ReLU(),
		decorators=[MD.BinomialDropout(0.2)],
		regularizations=[MR.L2(0.0001)], name="Hidden1")
	
	h2 = ML.Hidden(100,
		activation=MA.ReLU(),
		decorators=[MD.BinomialDropout(0.2)],
		regularizations=[MR.L1(0.0001)], name="Hidden2")
	
	outputLayer = ML.SoftmaxClassifier(10,
		learningScenario=ls,
		costObject=cost,
		name="OutputLayer",
		regularizations=[MR.L2(0.0002)])

	MLP = inputLayer > h1 > h2 > outputLayer
	
	#save html
	MLP.saveHTML("MLP_Example")

	# And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	#define maps
	trainData = MDM.Series(images = train_set[0], classes = train_set[1])
	trainMaps = MDM.DatasetMapper()
	trainMaps.mapInput(inputLayer, trainData.images)
	trainMaps.mapOutput(outputLayer, trainData.classes)

	testData = MDM.Series(images = test_set[0], classes = test_set[1])
	testMaps = MDM.DatasetMapper()
	testMaps.mapInput(inputLayer, testData.images)
	testMaps.mapOutput(outputLayer, testData.classes)

	validationData = MDM.Series(images = validation_set[0], classes = validation_set[1])
	validationMaps = MDM.DatasetMapper()
	validationMaps.mapInput(inputLayer, validationData.images)
	validationMaps.mapOutput(outputLayer, validationData.classes)

	#define trainer
	trainer = MT.DefaultTrainer(
		trainMaps=trainMaps,
		testMaps=testMaps,
		validationMaps=validationMaps,
		stopCriteria=[MSTOP.EpochWall(100)],
		trainMiniBatchSize=64
	)

	trainer.start("MLP_Example", MLP)
