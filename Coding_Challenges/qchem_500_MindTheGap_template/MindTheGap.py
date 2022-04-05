# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:55:16 2022

@author: digao
"""

import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.
    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian
    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    initial_state = np.array([1,1,0,0])
    dev = qml.device("default.qubit", wires=4)
    final_state = np.zeros(2**4, dtype = 'float64')
    
    def Circuit(params):
        qml.BasisState(initial_state, wires = range(4))
        qml.DoubleExcitation(params, wires=[0,1,2,3])
        
    @qml.qnode(dev)
    def Cost_H(params):
        Circuit(params)
        return qml.expval(H)
    
    optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0, requires_grad=True)
    energy = [Cost_H(theta)]
    max_ite = 100
    conv_tot = 1e-12
    
    for i in range(max_ite):
        theta, prev_energy = optimizer.step_and_cost(Cost_H, theta)
        
        energy.append(Cost_H(theta))
        conv = np.abs(energy[-1] - prev_energy)
        
        if conv <= conv_tot:
            break
        
    final_state[3] = -np.sin(theta/2)
    final_state[12] = np.cos(theta/2)
    
    return energy[-1], final_state
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.
    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()
    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #
    Hmat = qml.utils.sparse_hamiltonian(H, wires =[0,1,2,3])
    H_matrix = Hmat.toarray()
    proj_state = np.outer(ground_state, ground_state)
    H1_matrix = H_matrix + beta*proj_state
    
    return qml.Hermitian(H1_matrix, wires=[0,1,2,3])
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.
    Args:
        - H1 (qml.Observable): result of create_H1
    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    initial_state = np.array([1,1,0,0])
    dev = qml.device('default.qubit', wires = 4)
    
    def Circuit(params):
        qml.BasisState(initial_state, wires=range(4))
        qml.DoubleExcitation(params[0], wires=[0,1,2,3])
        for i in range(4):
            qml.SingleExcitation(params[i+1], wires = [i,(i+1)%4])
            
    @qml.qnode(dev)
    def Cost_H1(params):
        Circuit(params)
        return qml.expval(H1)
    
    optimizer = qml.AdamOptimizer(stepsize=0.1)
    theta = np.array([0,0.5,0.3,0.2,0.9], requires_grad=True, dtype = 'float64')
    energy = [Cost_H1(theta)]
    max_ite = 1000
    conv_tot = 1e-12
    
    for i in range(max_ite):
        theta, prev_energy = optimizer.step_and_cost(Cost_H1,theta)
        
        energy.append(Cost_H1(theta))
        conv = np.abs(energy[-1] - prev_energy)
        
        if conv <= conv_tot:
            break
        
    return energy[-1]
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")