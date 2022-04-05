# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:35:57 2022

@author: digao
"""

import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.
    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)
    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.
    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)
    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.
    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)
    Returns:
        - predictions (list(int)): Your final model predictions
    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires) 

    # Define a variational circuit below with your needed arguments and return something meaningful
    
    def hamiltonian(num_wires):
        coeffs = []
        obs = []
        for i in range(num_wires-1):
                coeffs.append(-1)
                obs.append(qml.PauliZ(i)@qml.PauliZ(i+1))
        return qml.Hamiltonian(coeffs, obs)
    
       
    def layer(W):
        
       for i in range(num_wires):
           qml.Rot(W[i,0],W[i,1],W[i,2], wires = i)
    
       for i in range(num_wires):
           qml.CNOT(wires = [i,(i+1)%num_wires])
        
        #for i in range(num_wires):
        #    qml.RY(W[i, 0], wires=i)
        #for i in range(num_wires):
        #    qml.CNOT(wires=[i, (i+1)%num_wires])
        #for i in range(num_wires):
        #    qml.RY(W[i, 1], wires=i)
        #for i in range(num_wires):
        #    qml.CNOT(wires=[i, (i+1)%num_wires])
        #for i in range(num_wires):
        #    qml.RY(W[i, 2], wires=i)
            
            
    def statepreparation(x):
        qml.BasisState(x, wires=range(num_wires))
    
    H = hamiltonian(num_wires)
    
    @qml.qnode(dev)
    def circuit(weights, x):
        
        statepreparation(x)
        
        for W in weights:
            layer(W)
        
        return qml.expval(H)
    
    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias
                
    # Define a cost function below with your needed arguments
    def cost(weights,bias, X, Y):

        # QHACK #
        
        # Insert an expression for your model predictions here
        predictions = [variational_classifier(weights, bias, x) for x in X]

        # QHACK #

        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here
    num_layers = 3
    weights_init = -0.001*np.random.randn(num_layers, num_wires, 3, requires_grad=True)
    print(weights_init)
    bias_init = np.array(-10.0, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(0.075)
    batch_size = 5
    
    weights = weights_init
    bias = bias_init
    
    for i in range(20):
        batch_index = np.random.randint(0, len(ising_configs), (batch_size,))
        X_batch = ising_configs[batch_index]
        Y_batch = labels[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)
        
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in ising_configs]
        acc = accuracy(labels, predictions)
    # QHACK #
    print(acc)
    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")