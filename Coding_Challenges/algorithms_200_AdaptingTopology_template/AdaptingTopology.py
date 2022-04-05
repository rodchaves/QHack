# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:16:13 2022

@author: digao
"""

#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.
    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'
    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    gate_wires = cnot.wires
    n = 0
    label = []
    target = gate_wires[1]
    
    
    if gate_wires[0] in graph[target]:
        return n
    
    def BFS(graph, root, n):
        q=[]
        label.append(root)
        q.append(root)
        depth_end = graph[root][-1]
        while q:
            discovered = q.pop(0)
            if discovered == target:
                return 2*n
            for i in graph[discovered]:
                if i not in label:
                    label.append(i)
                    q.append(i)
            if discovered == depth_end:
                n += 1
                depth_end = q[-1]
                    
    return BFS(graph, gate_wires[0], n)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")