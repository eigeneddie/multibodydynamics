import numpy        as np
import calcModule   as fc
import constraintModule as cmod
from   calcModule   import ATransformMatrixTHETA as A_Theta
from   calcModule   import ATransformMatrix as A_i
from   doublePendulu import mass_Matrix, constraintVect, mass2, mass3, gravity
from   doublePendulu import qiDot, u_bar_2A, u_bar_2B, u_bar_3B


def systemEquation (qi, t, Cq):
    
    massSize = mass_Matrix.shape[0]
    constVSize = constraintVect.shape[0]
    matDim =  massSize + constVSize
    massAugmented = np.zeros((matDim, matDim))
    massAugmented[0:massSize, 0:massSize] = mass_Matrix
    massAugmented[massSize:matDim, 0:massSize] = Cq
    massAugmented[0:massSize, massSize:matDim] = np.transpose(Cq)

    Qe = np.zeros((massSize,1), dtype = float)
    Qe[4] = -mass2*gravity
    Qe[7] = -mass3*gravity 
    
    Qd1 = cmod.QdCalc1(qi, qiDot, u_bar_2A, 2)
    Qd2 = cmod.QdCalc2(qi, qiDot, u_bar_2B, u_bar_3B, 2, 3)

    Qd = np.zeros((constVSize,1), dtype = float)
    Qd[0:2] = Qd1
    Qd[2:4] = Qd2

    QeAug = np.concatenate((Qe, Qd), axis = 0)
    mass_MatInverse = np.linalg.inv(massAugmented)
    qiDotDot_lamda = np.dot(mass_MatInverse, QeAug)

    return qiDotDot_lamda