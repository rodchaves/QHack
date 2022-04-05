# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:58:55 2022

@author: digao
"""

#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1, shots=1)


@qml.qnode(dev)
def is_bomb(angle):
    """Construct a circuit at implements a one shot measurement at the bomb.
    Args:
        - angle (float): transmissivity of the Beam splitter, corresponding
        to a rotation around the Y axis.
    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    qml.RY(2*angle, wires = 0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


@qml.qnode(dev)
def bomb_tester(angle):
    """Construct a circuit that implements a final one-shot measurement, given that the bomb does not explode
    Args:
        - angle (float): transmissivity of the Beam splitter right before the final detectors
    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    qml.PauliX(wires = 0)
    qml.RY(2*angle, wires = 0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


def simulate(angle, n):
    """Concatenate n bomb circuits and a final measurement, and return the results of 10000 one-shot measurements
    Args:
        - angle (float): transmissivity of all the beam splitters, taken to be identical.
        - n (int): number of bomb circuits concatenated
    Returns:
        - (float): number of bombs successfully tested / number of bombs that didn't explode.
    """

    # QHACK #
    
    D_beeps = 0
    boom = 0
    n_shots = int(np.ceil(10000/n))
    
    for i in range(n_shots):
        bomb_exploded = False
        for j in range(n):
            check = is_bomb(angle)
            if check == 1:
                boom += 1
                bomb_exploded = True
                break
                
        
        measurement = bomb_tester(angle)
        if measurement == 1 and not bomb_exploded:
            D_beeps += 1
                    
    return D_beeps/(n_shots-boom)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = simulate(float(inputs[0]), int(inputs[1]))
    print(f"{output}")