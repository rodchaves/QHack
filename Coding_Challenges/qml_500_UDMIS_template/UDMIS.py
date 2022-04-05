# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:28:52 2022

@author: digao
"""

import sys
import pennylane as qml
from pennylane import numpy as np


def hamiltonian_coeffs_and_obs(graph):
    """Creates an ordered list of coefficients and observables used to construct
    the UDMIS Hamiltonian.
    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]
    Returns:
        - coeffs (list): List of coefficients for elementary parts of the UDMIS Hamiltonian
        - obs (list(qml.ops)): List of qml.ops
    """

    num_vertices = len(graph)
    E, num_edges = edges(graph)
    u = 1.35
    obs = []
    coeffs = []

    # QHACK #
    # create the Hamiltonian coeffs and obs variables here
    for i in range(num_vertices):
        for j in range(i, num_vertices):
            if E[i][j] == True:
                coeffs.append(u/4)
                obs.append(qml.Identity(i)@qml.Identity(j))
                coeffs.append(u/4)
                obs.append(qml.PauliZ(i)@qml.Identity(j))
                coeffs.append(u/4)
                obs.append(qml.Identity(i)@qml.PauliZ(j))
                coeffs.append(u/4)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    for i in range(num_vertices):
        coeffs.append(-1/2)
        obs.append(qml.PauliZ(i))
        coeffs.append(-1/2)
        obs.append(qml.Identity(i))
    # QHACK #

    return coeffs, obs


def edges(graph):
    """Creates a matrix of bools that are interpreted as the existence/non-existence (True/False)
    of edges between vertices (i,j).
    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]
    Returns:
        - num_edges (int): The total number of edges in the graph
        - E (np.ndarray): A Matrix of edges
    """

    # DO NOT MODIFY anything in this code block
    num_vertices = len(graph)
    E = np.zeros((num_vertices, num_vertices), dtype=bool)
    for vertex_i in range(num_vertices - 1):
        xi, yi = graph[vertex_i]  # coordinates

        for vertex_j in range(vertex_i + 1, num_vertices):
            xj, yj = graph[vertex_j]  # coordinates
            dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            E[vertex_i, vertex_j] = 1 if dij <= 1.0 else 0

    return E, np.sum(E, axis=(0, 1))





def variational_circuit(params, num_vertices, H):
    """A variational circuit.
    Args:
        - params (np.ndarray): your variational parameters
        - num_vertices (int): The number of vertices in the graph. Also used for number of wires.
    """

    # QHACK #

    # create your variational circuit here
    def Mix_Hamiltonian(num_vertices):
         const = [1]*num_vertices
         op = []
         for i in range(num_vertices):
            op.append(qml.PauliX(i))
         H_mix = qml.Hamiltonian(const, op)
         return H_mix
    
    mix_H = Mix_Hamiltonian(num_vertices)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma,H)
        qml.qaoa.mixer_layer(alpha,mix_H)
   
    
    for i in range(num_vertices):
        qml.Hadamard(wires = i)
    qml.layer(qaoa_layer, 6, params[0], params[1])
     
    #for i in range(num_vertices):
    #    qml.RX(params[i], wires=i)
    #for i in range(num_vertices):
    #    for j in range(i,num_vertices):
    #        if i != j:
    #            qml.CNOT(wires=[i,j])
    #for i in range(num_vertices,2*num_vertices):
    #    qml.RX(params[i], wires=i%num_vertices)
    #for i in range(num_vertices):
    #    for j in range(i,num_vertices):
    #        if i != j:
    #            qml.CNOT(wires=[i,j])
    #for i in range(2*num_vertices,3*num_vertices):
    #    qml.RX(params[i], wires=i%num_vertices)
    # QHACK #


def train_circuit(num_vertices, H):
    """Trains a quantum circuit to learn the ground state of the UDMIS Hamiltonian.
    Args:
        - num_vertices (int): The number of vertices/wires in the graph
        - H (qml.Hamiltonian): The result of qml.Hamiltonian(coeffs, obs)
    Returns:
        - E / num_vertices (float): The ground state energy density.
    """

    dev = qml.device("default.qubit", wires=num_vertices)
    
    @qml.qnode(dev)
    def cost(params):
        """The energy expectation value of a Hamiltonian"""
        variational_circuit(params, num_vertices, H)
        return qml.expval(H)

    # QHACK #

    # define your trainable parameters and optimizer here
    opt = qml.AdagradOptimizer(stepsize=0.3555)
    params = np.array(np.ones((2,6))*0.3, requires_grad=True)
    #params = np.array(np.ones(num_vertices*4)*0.5, requires_grad=True)
    # change the number of training iterations, `epochs`, if you want to
    # just be aware of the 80s time limit!
    
    epochs = 300

    # QHACK #

    for i in range(epochs):
        params = opt.step(cost, params)
        
    E = cost(params)
    return E / float(num_vertices)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float, requires_grad=False)
    num_vertices = int(len(inputs) / 2)
    x = inputs[:num_vertices]
    y = inputs[num_vertices:]
    graph = []
    for n in range(num_vertices):
        graph.append((x[n].item(), y[n].item()))

    coeffs, obs = hamiltonian_coeffs_and_obs(graph)
    H = qml.Hamiltonian(coeffs, obs)

    energy_density = train_circuit(num_vertices, H)
    print(f"{energy_density:.6f}")