# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:45:59 2022

@author: digao
"""

#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.
    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.
    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    dev = qml.device('default.qubit', wires = 8, shots = 1)
    
    def oracle():
        fs[0]([0,1,4])
        fs[1]([2,3,5])
        fs[2]([0,1,6])
        fs[3]([2,3,7])
    
    @qml.qnode(dev)
    def circuit():
        for i in range(4,8):
            qml.PauliX(wires = i)
        for i in range(8):
            qml.Hadamard(wires = i)
        oracle()
        for i in range(4):
            qml.Hadamard(wires = i)
    
        return qml.sample(wires = range(4))
    
    probs = circuit()
    check = 0
    for i in range(4):
        check += probs[i]
    
    if check == 0:
        return '4 same'
    else:
        return '2 and 2'
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")