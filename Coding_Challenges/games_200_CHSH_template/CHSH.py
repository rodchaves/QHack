# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:46:26 2022

@author: digao
"""

#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.
    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    factor = abs(alpha)**2+abs(beta)**2
    alpha = alpha/np.sqrt(factor)
    beta = beta/np.sqrt(factor)
    state = [alpha, 0, 0, beta]
    qml.QubitStateVector(state,wires=[0,1])
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases
    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK # 
    
    #for i in range(2):
    #    qml.Hadamard(wires = i)
    #qml.CNOT(wires = [0,1])
    #qml.adjoint(qml.RZ)(2*theta_A0, wires = 0)
    #qml.adjoint(qml.RZ)(2*theta_A1, wires = 1)
    #qml.adjoint(qml.RX)(2*theta_B0, wires = 0)
    #qml.adjoint(qml.RX)(2*theta_B1, wires = 1)
    #qml.CNOT(wires = [1,0])
    
    if x == 0 and y == 0:
    #    qml.Hadamard(wires = 0)
        qml.adjoint(qml.RY)(2*theta_A0, wires = 0)
    #    qml.Hadamard(wires = 1)
        qml.adjoint(qml.RY)(2*theta_B0, wires = 1)
    elif x == 0 and y == 1:
    #    qml.Hadamard(wires = 0)
        qml.adjoint(qml.RY)(2*theta_A0, wires = 0)
    #    qml.Hadamard(wires = 1)
        qml.adjoint(qml.RY)(2*theta_B1, wires = 1)
    elif x == 1 and y == 0:
    #    qml.Hadamard(wires = 0)
        qml.adjoint(qml.RY)(2*theta_A1, wires = 0)
    #    qml.Hadamard(wires = 1)
        qml.adjoint(qml.RY)(2*theta_B0, wires = 1)       
    else:
    #    qml.Hadamard(wires = 0)
        qml.adjoint(qml.RY)(2*theta_A1, wires = 0)
    #    qml.Hadamard(wires = 1)
        qml.adjoint(qml.RY)(2*theta_B1, wires = 1)
       
    # QHACK #
    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.
    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    
    win_prob = 0
    x = np.random.randint(2, size = 1)
    y = np.random.randint(2, size = 1)
    target = x*y
    
    probs_game = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
    
    if target == 0:
        win_prob = probs_game[0]+probs_game[3]
    else:
        win_prob = probs_game[1]+probs_game[2]
    
    return win_prob
               
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game
    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""

    # QHACK #
        loss = 0
        points = 100
        for i in range(points):
            loss = loss + (1-winning_prob(params,alpha,beta))**2
            #loss += -np.log(winning_prob(params,alpha,beta))
            #loss += np.log(1+np.exp(winning_prob(params,alpha,beta)))/np.log(2)
            #loss += 1/(1+np.exp(winning_prob(params,alpha,beta)))**2
            #loss += abs(1-winning_prob(params,alpha,beta))
        return loss/points
    
    
    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi], requires_grad = True)
    print(init_params)
    opt = qml.AdagradOptimizer(stepsize = 0.05)
    steps = 200

    # QHACK #
    
    # set the initial parameter values
    params = init_params
 
    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost, params)
    
        
        # QHACK #
    print(params)    
    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")