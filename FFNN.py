import numpy as np
import pandas as pd

def readFromFile(path):
    f = open(path, "r")

    line1 = f.readline()
    trainingExamples = int(f.readline())
    # f.close
    inputList = line1.split()
    inputNodesNumber = int(inputList.__getitem__(0))
    inputHiddenNumber = int(inputList.__getitem__(1))
    outputNodesNumber = int(inputList.__getitem__(2))

    count = 0
    df = pd.DataFrame()
    while count < trainingExamples:
        count += 1

        # Get next line from file
        line = f.readline().split()
        # [float(i) for i in line]
        line2 = [x.strip(' ') for x in line]
        # data.append(line)
        df = df.append(pd.Series(line2), ignore_index=True)

    df.columns.astype(float)

    X = df.iloc[0:trainingExamples, 0:inputNodesNumber]
    y = df.iloc[0:trainingExamples, inputNodesNumber - outputNodesNumber + 1:]

    X = np.array(X.values)
    y = np.array(y.values)



    X = X.astype(float)
    y = y.astype(float)

    np.seterr(divide='ignore', invalid='ignore')  # there is no error but there is :) "dividing by zero" now there is no

    # normalization

    for index in range(inputNodesNumber):

        X[:, index] = (X[:, index] - X[:, index].mean(axis=0)) / X[:, index].std(axis=0)

    np.set_printoptions(suppress=True)


    return inputNodesNumber, inputHiddenNumber, outputNodesNumber, X, y


def writeToFile(path, finalWeights1, finalWeights2):
    file = open(path, "w+")

    content = str(finalWeights1)
    file.write(content)
    file.write("\n \n \n \n")
    content = str(finalWeights2)
    file.write(content)

    file.close()


# Neural Network begin here
def sigmoid_derivative(x):
    return x * (1.0 - x)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def meanSquareError(y, yPred):
    return np.mean((y - yPred) ** 2)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # print("X \n", x)
        self.weights1 = np.random.rand(self.input.shape[1], inputHiddenNumber)
        # print("weight1 \n", self.weights1)
        self.weights2 = np.random.rand(inputHiddenNumber, 1)
        # print("weight2 \n", self.weights2)
        self.y = y
        # print("y\n",y)
        self.output = np.zeros(self.y.shape)  # expectid y
        # print("output\n",self.ouput)

    # getter method
    def get_weights1(self):
        return self.weights1

    def get_weights2(self):
        return self.weights2

    # setter method
    def set_weights1(self, w1):
        self.weights1 = w1

    def set_weights2(self, w2):
        self.weights2 = w2

    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # print("layer 1 \n", self.layer1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        # print("layer 2 (output) \n", self.output)

    def backprop(self):
        b_weight2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))

        # print("shapes of y = \n ", self.y.shape)
        b_weight1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1)
                                          )
                           )

        self.weights1 += b_weight1
        self.weights2 += b_weight2


# -----------------------------------------
path = 'D:\\Uni\\FCI_Y4_T1\\FCI_Y4_T1\\Genetic\\Assignments\\A4\\20170113-20170419-CS-DS-G1 (A4)\\train.txt'

inputNodesNumber, inputHiddenNumber, outputNodesNumber, X, y = readFromFile(path)

nn = NeuralNetwork(X, y)

for i in range(500):
    nn.feedForward()
    nn.backprop()
print("output \n", nn.output)

mse = meanSquareError(nn.output, y)
print("mse", mse)

returnedWeight1 = nn.get_weights1()
returnedWeight2 = nn.get_weights2()

# write weights to file
finalWeightsFile = 'D:\\Uni\\FCI_Y4_T1\\FCI_Y4_T1\\Genetic\\Assignments\\A4\\20170113-20170419-CS-DS-G1 (A4)\\test.txt'

writeToFile(finalWeightsFile, returnedWeight1, returnedWeight2)

# ---------------------- test--------------------
path2 = 'D:\\Uni\\FCI_Y4_T1\\FCI_Y4_T1\\Genetic\\Assignments\\A4\\20170113-20170419-CS-DS-G1 (A4)\\train2.txt'
inputNodesNumber2, inputHiddenNumber2, outputNodesNumber2, X2, y2 = readFromFile(path2)

test = NeuralNetwork(X2, y2)

test.set_weights1(returnedWeight1)
test.set_weights2(returnedWeight2)

test.feedForward()

mse2 = meanSquareError(test.output, y2)
print("mse2 = ", mse2)
