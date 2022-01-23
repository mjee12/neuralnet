import sys
import numpy as np
from numpy import genfromtxt
import math 

def openFile(data):
  data = genfromtxt(data, delimiter = ",")
  labels = data[:,0]
  data = data[:,1:]
  #data2 = data[0] #temporary for debugging
  dataT = data.transpose()
  add = np.ones(len(labels))
  dataT = np.vstack((dataT, add))
  return labels, dataT

#convert labels to one hot
def oneHot(labels, outputSize, trainNum):
  result = np.zeros((outputSize, trainNum))
  for i, elem in enumerate(labels):
    result[int(elem), i] = 1
  return result.T

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def softmax(x):
  top = np.exp(x)
  bottom = np.sum(top, axis=0, keepdims=True) 
  return top / bottom

def forward(data, w1, w2, labels):
  a = np.dot(data, w1)
  z = sigmoid(a)
  z2 = np.append(z, 1)
  b = np.dot(z2, w2)
  yHat = softmax(b)
  return a, z2, b, yHat

def crossEntropy(y, yHat):
  return -np.matmul(y,np.log(yHat))

def derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

def backward(data, labels, w2, z2, yHat):
  g_b = yHat - labels.T
  g_beta = np.outer(z2, g_b)
  g_z = np.matmul(w2, g_b)
  g_z = g_z[:-1]
  z = z2[:-1]
  g_a = g_z * (z - (np.multiply(z, z)))
  g_alpha = np.outer(data, g_a)
  return g_beta, g_alpha

def sgd(data, data2, w1, w2, labels2, testLabels2, learnR, numEpoch, trainNum, testNum):
  trainEntropy = {}
  testEntropy = {}
  for i in range(numEpoch):
    train = 0
    test = 0
    for j in range(trainNum):
      dataT = data[:,j]
      lab = labels2[j]
      a, z2, b, yHat = forward(dataT, w1, w2, lab)
      train += crossEntropy(lab, yHat)
      g_beta, g_alpha = backward(dataT, lab, w2, z2, yHat)
      w1 = w1 - np.multiply(learnR, g_alpha)
      w2 = w2 - np.multiply(learnR, g_beta)
    trainEntropy[i+1] = train / float(trainNum)

    for k in range(testNum):
      dataT2 = data2[:,k]
      testLab = testLabels2[k]
      a, z2, b, yHat = forward(dataT2, w1, w2, testLab)
      test += crossEntropy(testLab, yHat)
    testEntropy[i+1] = test / float(testNum)

  return w1, w2, trainEntropy, testEntropy

def predict(data, w1, w2, labels):
  count = 0
  predict = []
  num = labels.shape[0]
  for i in range(num):
    dataT = data[:,i]
    lab = labels[i]
    a, z2, b, yHat = forward(dataT, w1, w2, lab)
    pred = np.argmax(yHat)
    if pred != lab:
      count += 1
    predict.append(pred)
  return predict, count

def main():
  trainInput = sys.argv[1]
  testInput = sys.argv[2]
  trainOut = sys.argv[3]
  testOut = sys.argv[4]
  metricsOut = sys.argv[5]
  numEpoch = int(sys.argv[6])
  hiddenVal = int(sys.argv[7])
  initVal = int(sys.argv[8])
  learnR = float(sys.argv[9])

  labels, dataT = openFile(trainInput)
  testLabels, testDataT = openFile(testInput)

  #initialization of variables
  trainNum = labels.shape[0]
  testNum = testLabels.shape[0]
  featNum = dataT.shape[0]-1
  outputSize = 10

  #redefine labels
  labels2 = oneHot(labels, outputSize, trainNum) 
  testLabels2 = oneHot(testLabels, outputSize, testNum)

  #initializing weights and bias
  if initVal == 1:
    w1 = np.random.uniform(-0.1, 0.1, (featNum+1, hiddenVal))
    w2 = np.random.uniform(-0.1, 0.1, (hiddenVal+1, outputSize))
  if initVal == 2:
    w1 = np.zeros((featNum+1, hiddenVal))
    w2 = np.zeros((hiddenVal+1, outputSize))
  
  finalW1, finalW2, trainEntropy, testEntropy \
     = sgd(dataT, testDataT, w1, w2, labels2, testLabels2, learnR, numEpoch, trainNum, testNum)

  trainOutput, trainCount = predict(dataT, finalW1, finalW2, labels)
  testOutput, testCount = predict(testDataT, finalW1, finalW2, testLabels)

  trainError = trainCount / float(trainNum)
  testError = testCount / float(trainNum)

  with open(trainOut, "w") as file:
    for i in trainOutput:
      file.write("%d\n" % i)
  file.close()

  with open(testOut, "w") as file:
    for i in testOutput:
      file.write("%d\n" % i)
  file.close()

  with open(metricsOut, "w") as file:
    num = trainEntropy.keys()
    for i in num:
      file.write("epoch=%d crossentropy(train): %f\n" % (i, trainEntropy[i]))
      file.write("epoch=%d crossentropy(test): %f\n" % (i, testEntropy[i]))
    file.write("error(train): %f\n" % (trainError))
    file.write("error(test): %f" % (testError))
  file.close()

if __name__ == "__main__":
  main()