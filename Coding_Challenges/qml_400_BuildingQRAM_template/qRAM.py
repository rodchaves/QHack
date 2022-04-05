# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:33:44 2022

@author: digao
"""

#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.
    Args:
        - thetas (list(float)): list of angles to apply in the rotations.
    Returns:
        - (list(complex)): final state.
    """

    # QHACK #
    # Use this space to create auxiliary functions if you need it.
  
    
    def Memory(theta, control):
        bits = []
        bitstring = np.binary_repr(control, width = 3)
        for i in range(-1,-4,-1):
            if int(bitstring[i]) == 0:
                bits.append(i+3)
        for i in range(len(bits)):
            qml.PauliX(wires = bits[i])
        qml.ctrl(qml.RY, control = [0,1,2])(theta, wires = 3)
        for i in range(len(bits)):
            qml.PauliX(wires = bits[i])

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        for i in range(3):
           qml.Hadamard(wires = i)
         
        for i in range(3):
            qml.PauliX(wires = i)
        qml.ctrl(qml.RY, control = [2,1,0])(thetas[0], wires = 3)
        for i in range(3):
            qml.PauliX(wires = i)
        
        for i in range(1,len(thetas)):
            Memory(thetas[i], i)
        
        
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")