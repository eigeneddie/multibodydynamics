#hard coded for double pendulum dynamic system
import numpy        as np
import calcModule   as fc
from   calcModule   import ATransformMatrixTHETA as A_Theta, link2index
from   calcModule   import ATransformMatrix      as A_i

def constraintEquation(r1A, r1B, r2B): #, r3C, timeNow):
    
    constraintVector = np.zeros((4,1))

    # Pin joint A
    constraintPinA = -r1A 
    for i in range(np.size(constraintPinA)):
        # Equation 1-2
        constraintVector[i] = constraintPinA[i]

    # Pin joint B
    constraintPinB = r1B-r2B
    for i in range(np.size(constraintPinB)):
        # Equation 3-4
        constraintVector[i+2] = constraintPinB[i]

    return constraintVector

def jacobianMatrix(qi, u_bar_1A, u_bar_1B, u_bar_2B):
    
    genCoor = 6 # number of generalized coordinates
    constEq = 4 # number of constraint equations

    jacobianMatrixCq = np.zeros((constEq, genCoor))
    identity2x2 = np.identity(2)

    # row 1-2
    Cq12 = np.dot(A_Theta(qi[link2index(1,"theta")]), u_bar_1A)
    jacobianMatrixCq[0:2,0:2] = -identity2x2
    jacobianMatrixCq[0:2,2:3] = -Cq12

    # row 3-4 (r2A = r3A)
    Cq34_link2 = np.dot(A_Theta(qi[link2index(1,"theta")]), u_bar_1B)
    Cq34_link3 = np.dot(A_Theta(qi[link2index(2,"theta")]), u_bar_2B)

    jacobianMatrixCq[2:4,0:2] = identity2x2
    jacobianMatrixCq[2:4,2:3] = Cq34_link2
    jacobianMatrixCq[2:4,3:5] = -identity2x2
    jacobianMatrixCq[2:4,5:6] = -Cq34_link3
    #print(np.size(Cq34_link3))

    # SLICING
    # a. jacobian dependent
    jacobian_dependent = np.concatenate((jacobianMatrixCq[:, 0:2], 
                                         jacobianMatrixCq[:, 3:5]), axis = 1)
    # b. jacobian independent
    jacobian_independent = np.concatenate((jacobianMatrixCq[:, 2:3], 
                                           jacobianMatrixCq[:, 5:6]), axis = 1)

    return jacobianMatrixCq, jacobian_dependent, jacobian_independent

def constrEqDot(constraintVectorVel, omega):

    constraintVectorVel[11] = -omega

    return constraintVectorVel

def positionAnalysis(constraintVector, jacobianMatrix, qi):
    inverse_jacobian = np.linalg.inv(jacobianMatrix)
    delta_qi = - np.matmul(inverse_jacobian, constraintVector)
    delta_qi_norm = np.linalg.norm(delta_qi)
    qi = qi + delta_qi

    return qi, delta_qi_norm

def velocityAnalysis(constraintVectorVel, jacobianMatrix, qiDot):
    inverse_jacobian = np.linalg.inv(jacobianMatrix)
    qiDot= - np.matmul(inverse_jacobian, constraintVectorVel)

    return qiDot

def accelerationAnalysis(jacobianMatrix, qiDotDot, qiDot, qi,
                        u_bar_2O, u_bar_2A, u_bar_3A, u_bar_3B, u_bar_4B,
                        u_bar_1O4, u_bar_4O4):
    # Qd line 4-5                    
    Qd45 = np.square(float(qiDot[5]))*np.dot(fc.ATransformMatrix(float(qi[5])), u_bar_2O)
    
    # Qd line 6-7
    Qd67a = np.square(float(qiDot[5]))*np.dot(fc.ATransformMatrix(float(qi[5])), u_bar_2A)
    Qd67b = np.square(float(qiDot[8]))*np.dot(fc.ATransformMatrix(float(qi[8])), u_bar_3A)
    Qd67 = Qd67a-Qd67b

    # Qd line 8-9
    Qd89a = np.square(float(qi[8]))*np.dot(fc.ATransformMatrix(float(qi[8])), u_bar_3B)
    Qd89b = np.square(float(qi[11]))*np.dot(fc.ATransformMatrix(float(qi[11])), u_bar_4B)
    Qd89 = Qd89a-Qd89b

    # Qd line 10-11
    Qd1011a = np.square(float(qi[2]))*np.dot(fc.ATransformMatrix(float(qi[2])), u_bar_1O4)
    Qd1011b = np.square(float(qi[11]))*np.dot(fc.ATransformMatrix(float(qi[11])), u_bar_4O4)
    Qd1011 = Qd1011a-Qd1011b


    Qd = np.array([[0],[0],[0], Qd45[0], Qd45[1],
                    Qd67[0], Qd67[1], Qd89[0], Qd89[1], 
                    Qd1011[0], Qd1011[1], [0]])
    
    inverse_jacobian = np.linalg.inv(jacobianMatrix)
    qiDotDot= np.dot(inverse_jacobian, Qd)

    return qiDotDot

def QdCalc1(qi, qiDot, u_bar_iP, i):
    id = link2index(i, "theta")
    Qd = np.square(float(qiDot[id]))*np.dot(A_i(qi[id]), u_bar_iP)
    return Qd 

def QdCalc2(qi, qiDot, u_bar_iP, u_bar_jP, i, j): 
    id = link2index(i, "theta")
    jd = link2index(j, "theta")
    Qda = np.square(float(qiDot[id]))*np.dot(A_i(qi[id]), u_bar_iP)
    Qdb = np.square(float(qiDot[jd]))*np.dot(A_i(qi[jd]), u_bar_jP)
    Qd = Qda-Qdb
    return Qd
