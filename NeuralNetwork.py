import numpy
import scipy.special



class Network:

    def __init__(self, innode, outnode, hidnode, learningrate):
        self.iNode = innode
        self.oNode = outnode
        self.hNode = hidnode
        self.lr = learningrate
        #Weights
        self.wih = numpy.random.normal(0.0, pow(self.iNode, -0.5), (self.hNode, self.iNode))
        self.who = numpy.random.normal(0.0, pow(self.hNode, -0.5), (self.oNode, self.hNode))

        self.activationfunction = lambda x: scipy.special.expit(x)

    def learn(self, inputs, targets):
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T
        
        #each node applys math (weights value and sigmoid function(activation)) for  all of the hidden layers
        hidInputs = numpy.dot(self.wih, inputs)
        #sigmoid
        hidOutputs = self.activationfunction(hidInputs)

        #Taking outputs from hidden layer(s) and applying to output layer
        finInputs = numpy.dot(self.who, hidOutputs)
        #sigmoid
        finOutputs = self.activationfunction(finInputs)

        #Calculation error and back propogation
        outerror = targets - finOutputs
        #numpydotdot calcs hidden layer error, which is just outerror, split by weights and recombined at hidden nodes
        hiderror = numpy.dot(self.who.T, outerror)

        #now we can update weights
        self.who += self.lr * numpy.dot((outerror * finOutputs * (1.0 - finOutputs)),numpy.transpose(hidOutputs))
        self.wih += self.lr * numpy.dot((outerror * hidOutputs * (1.0 - hidOutputs)),numpy.transpose(inputs))
        
        pass
    
    def predict(self, inputs):
        #convert inputs to array
        inputs = numpy.array(inputs, ndmin = 2).T

        #run through hidden layer(s)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activationfunction(hidden_inputs)

        #Output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        final_prediction = self.activationfunction(final_inputs)
        return final_prediction
        

learningrate = 0.1
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

n = Network(input_nodes,hidden_nodes,output_nodes, learningrate)






training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        print(all_values)
        #print(all_values)
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.learn(inputs, targets)
        pass
    pass

test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.predict(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    print(str(label) + " / " + str(correct_label))
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)










