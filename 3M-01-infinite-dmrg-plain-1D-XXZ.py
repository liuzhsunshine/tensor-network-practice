#!/usr/bin/env python
# Modification from the following:
# Simple DMRG example for 1D XXZ model
#Ian McCulloch, August 2017
# <https://people.smp.uq.edu.au/IanMcCulloch/mptoolkit/index.php?n=Tutorials.SimpleDMRG?from=APCTPWorkshop.SimpleDMRG>

import numpy as np
import scipy
import scipy.sparse.linalg
import math

import matplotlib.pyplot as plt
import time


####Initial parameter
#physical parameter
J = 1
Jz = 1
#number of states kept
m = 10
#number of iterations
NIter = 100

####Open interactive mode
plt.ion()
plt.figure(1)

#exact solution
ExactEnergy = -math.log(2) + 0.25
print("    Iter    Size     Energy      BondEnergy     EnergyError   Truncation")

####Local operator
I = np.mat(np.identity(2))
Sz = np.mat([[0.5, 0],
            [0, -0.5]])
Sp = np.mat([[0, 1],
             [0, 0]])
Sm = np.mat([[0, 0],
             [1, 0]])
Zero = np.zeros((2, 2))

####Initial block operators
####Here we assume symmetric reflections

BlockSz = Sz
BlockSp = Sp
BlockSm = Sm
BlockI = I
BlockH = np.zeros((2, 2))

Energy = -0.75

####Begin main iteration
for i in range(0, NIter):

    #Create an enlarged block
    BlockH = np.kron(BlockH, I) + \
             Jz * np.kron(BlockSz, Sz) + 0.5 * J * (np.kron(BlockSp, Sm) + np.kron(BlockSm, Sp)) + \
             np.kron(BlockI, Zero)
    BlockSz = np.kron(BlockI, Sz)
    BlockSm = np.kron(BlockI, Sm)
    BlockSp = np.kron(BlockI, Sp)
    BlockI = np.kron(BlockI, I)

    #Create superblock Hamiltonian
    H_super = np.kron(BlockH, BlockI) + \
              Jz *np.kron(BlockSz, BlockSz) + 0.5 * J * (np.kron(BlockSp, BlockSm) + np.kron(BlockSm, BlockSp)) + \
              np.kron(BlockI, BlockH)

    H_super = 0.5 * (H_super + H_super.H)

    #Diagonalze the Hamiltonian
    LastEnergy = Energy
    E, Psi = scipy.sparse.linalg.eigsh(H_super, k=1, which='SA')
    Energy=E[0]
    EnergyPerBond = (Energy - LastEnergy)/2

    #Form reduced density matrix
    Dim = BlockH.shape[0]
    PsiMatrix = np.mat(np.reshape(Psi, [Dim, Dim]))
    Rho =  PsiMatrix.H * PsiMatrix

    #Diagonalize the density matrix
    D, V = np.linalg.eigh(Rho)

    #Construct projector
    T = np.mat(V[:, max(0, Dim-m):Dim])
    TruncationError = 1-np.sum(D[max(0, Dim-m):Dim])

    print("{:6} {:6} {:16.8f} {:12.8f} {:12.8f} {:12.8f}".format(i, 4 + i * 2, Energy, EnergyPerBond, ExactEnergy-EnergyPerBond, TruncationError))

    #Restore the Hamiltonian
    BlockH = T.H * BlockH * T
    BlockSz = T.H * BlockSz * T
    BlockSp = T.H * BlockSp * T
    BlockSm = T.H * BlockSm * T
    BlockI = T.H * BlockI * T

    t_now = i * 0.1
    plt.scatter(t_now, Energy)
    plt.pause(0.01)


print("Finished")