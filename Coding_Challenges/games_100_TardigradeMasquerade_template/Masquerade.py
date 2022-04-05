# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:10:22 2022

@author: digao
"""

import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.
    Args:
        - theta (float): the angle that defines the state psi_ABT
    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    # QHACK #
    @qml.qnode(dev)
    def No_Tardigrade():
        qml.Hadamard(wires = 0)
        qml.PauliX(wires = 0)
        qml.CNOT(wires = [0,1])
        qml.PauliX(wires = 0)
        
        return qml.density_matrix([1])
    
    @qml.qnode(dev)
    def Tardigrade(theta):
        qml.Hadamard(wires = 0)
        qml.CRY(theta, wires =[0,1])
        qml.CNOT(wires = [1,2])
        qml.CNOT(wires=[0,1])
        qml.PauliX(wires = 0)
        
        return qml.density_matrix([1])
    
    S2_without = second_renyi_entropy(No_Tardigrade())
    S2_with = second_renyi_entropy(Tardigrade(theta))
    
    
    return S2_without, S2_with
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")