import numpy as np
from random import choice

# Dataset XOR -> Untuk training kita
dataset = [
    (np.array([0,0]), np.array([0])),
    (np.array([0,1]), np.array([1])),
    (np.array([1,0]), np.array([1])),
    (np.array([1,1]), np.array([0]))
]

# Inputnya dia minta 2, dan menghasilkan 1 output
num_of_input = 2
num_of_output = 1

# Awalnya weightnya random karena kita gatau apa apa
weight = np.random.normal(size=[num_of_input, num_of_output])
# Bias itu penambahannya -> gangguan dari luar
bias = np.random.normal(size=[num_of_output])

def activation(output):
    # Pelabelan untuk ANN - maksudnya apah?
    if(output >= 0):
        return 1
    return 0

def feed_forward(feature):
    # Feature [0, 0]
    # Weight 
    # [1,
    #  2]
    # print(feature)
    # print(weight)
    output = np.matmul(feature, weight) + bias
    return activation(output)

epoch = 1000
# Seberapa cepet modelnya harus belajar
learning_rate = 0.1

for i in range(epoch):
    feature, target = choice(dataset)

    output = feed_forward(feature)
    error = target - output
    
    weight = weight + learning_rate * error * feature.reshape(2, 1)
    bias = bias + learning_rate * error

    if(i % 20 == 0):
        correct = 0
        for feature, target in dataset:
            output = feed_forward(feature)
            if(target == output):
                correct += 1
        print("Epoch: ", i , " | Accuracy: ", correct/len(dataset)*100,"%")
    

    
