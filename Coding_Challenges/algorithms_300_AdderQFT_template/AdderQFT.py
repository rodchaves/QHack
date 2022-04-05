# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:36:41 2022

@author: digao
"""

#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qfunc_adder(m, wires):
    """Quantum function capable of adding m units to a basic state given as input.
    Args:
        - m (int): units to add.
        - wires (list(int)): list of wires in which the function will be executed on.
    """

    qml.QFT(wires=wires)

    # QHACK #
    bitstring = np.binary_repr(m, width=wires[-1]+1)
    for i in range(wires[-1],-1,-1):
        k = i
        for j in range(wires[-1], wires[-1]-i-1, -1):
            if bitstring[j] == '1':
                qml.PhaseShift(2*np.pi/(2**(k+1)), wires = i)     
    
            k -= 1    
    # QHACK #

    qml.QFT(wires=wires).inv()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    m = int(inputs[0])
    n_wires = int(inputs[1])
    wires = range(n_wires)

    dev = qml.device("default.qubit", wires=wires, shots=1)

    @qml.qnode(dev)
    def test_circuit():
        # Input:  |2^{N-1}>
        qml.PauliX(wires=0)

        qfunc_adder(m, wires)
        return qml.sample()

    output = test_circuit()
    print(*output, sep=",")