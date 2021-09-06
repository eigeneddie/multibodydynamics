# Copyright (C) 2021 Edgar Sutawika - All Rights Reserve
# For educational purposes

#equations for constraints
import numpy        as np
from   calcModule3D   import link2index as l2i
from   calcModule3D   import ATrans     as A_i
from numpy import square as sq 

def constraintEquation(qi, i):
    id = l2i(i, "t0")
    t0, t1, t2, t3 = qi[id], qi[id+1], qi[id+2], qi[id+3]
    constraintVector = sq(t0) + sq(t1) + sq(t2) + sq(t3) - 1

    return constraintVector

def jacobianMatrix(qi, i):
    id = l2i(i, "t0")
    t0, t1, t2, t3 = qi[id], qi[id+1], qi[id+2], qi[id+3]
    Cq = np.array([[0, 0, 0, 2*t0, 2*t1, 2*t2, 2*t3]])

    return Cq

def QdEulPar1(qi, i):
    id = l2i(i, "t0")
    t0, t1, t2, t3 = qi[id], qi[id+1], qi[id+2], qi[id+3]
    Qd_EP1 = -(2*t0**2 + 2*t1**1 + 2*t2**2 + 2*t3**2)

    return Qd_EP1 

def positionAnalysis(constraintVector, jacobianMatrix, qi):
    inverse_jacobian = np.linalg.inv(jacobianMatrix)
    delta_qi = - np.matmul(inverse_jacobian, constraintVector)
    delta_qi_norm = np.linalg.norm(delta_qi)
    qi = qi + delta_qi

    return qi, delta_qi_norm

def QdJoint1(qi, qiDot, u_bar_iP, i):
    id = l2i(i, "theta")
    Qd_RJ1 = np.square(float(qiDot[id]))*np.dot(A_i(qi[id]), u_bar_iP)
    return Qd_RJ1 

def QdJoint2(qi, qiDot, u_bar_iP, u_bar_jP, i, j): 
    id = l2i(i, "theta")
    jd = l2i(j, "theta")
    Qda = np.square(float(qiDot[id]))*np.dot(A_i(qi[id]), u_bar_iP)
    Qdb = np.square(float(qiDot[jd]))*np.dot(A_i(qi[jd]), u_bar_jP)
    Qd_RJ2 = Qda-Qdb
    return Qd_RJ2

def revolutJoint (riP, riJ):
    constraintPin = riP-riJ
    
    return constraintPin