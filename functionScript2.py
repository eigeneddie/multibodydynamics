# Custom Functions
import numpy as np
import pandas as pd

def ATransformMatrix (theta): #A
    ATransform = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
    return ATransform 

def ATransformMatrixTHETA (theta): #A_theta
    ATransformTHETA = np.array([[-np.sin(theta), -np.cos(theta)], 
                                [np.cos(theta), -np.sin(theta)]])
    return ATransformTHETA

def prettyMatVect(matVect):
    prettyMatVect = pd.DataFrame(matVect, columns =
                                ['R1x', 'R1y', 'theta1',
                                'R2x','R2y','theta2',
                                'R3x','R3y','theta3',
                                'R4x','R4y','theta4'])
    return prettyMatVect

def calcGlobalCoor(qi, u_bar_iP, u_bar_jP, link_i, link_j):
    # To calculate Point of Interest positions in terms of global coordinates
    index_for_i = 3*(link_i-1)
    index_for_j = 3*(link_j-1)
    Ri = np.array([qi[index_for_i], qi[index_for_i+1]], dtype = float) 
    Rj = np.array([qi[index_for_j], qi[index_for_j+1]], dtype = float)

    # Point P at link i
    riP = Ri + np.matmul(ATransformMatrix(float(qi[index_for_i+2])), u_bar_iP)
    # Point P at link j
    rjP = Rj + np.matmul(ATransformMatrix(float(qi[index_for_j+2])), u_bar_jP)

    return riP, rjP

def constraintEquation(constraintVector, qi, r1O2, r2O2, r2A, r3A, 
                    r3B, r4B, r1O4, r4O4, 
                    theta2Initial, omega2, timeNow):
    # Ground constraint
    for i in range(3):
        constraintVector[i] = 0 #qi[i] # just zero

    # Pin joint O2
    constraintPinO = r2O2# r1O -
    for i in range(np.size(constraintPinO)):
    # Equation 4-5
        constraintVector[i+3] = constraintPinO[i]

    # Pin joint A
    constraintPinA = r2A - r3A
    for i in range(np.size(constraintPinA)):
    # Equation 6-7
        constraintVector[i+5] = constraintPinA[i]

    # Pin joint B
    constraintPinB = r3B - r4B
    for i in range(np.size(constraintPinB)):
    # Equation 8-9
        constraintVector[i+7] = constraintPinB[i]

    # Pin joint O4
    constraintPinO4 = r1O4-r4O4
    for i in range(np.size(constraintPinO4)):
    # Equation 10-11
        constraintVector[i+9] = constraintPinB[i]

    # Equation 12 ===> Driving constraint
    constraintVector[11] = qi[5] - theta2Initial - omega2*timeNow 

    return constraintVector

def jacobianMatrix(qi, u_bar_2O2, u_bar_2A, u_bar_3A, u_bar_3B, 
                    u_bar_4B, u_bar_1O4, u_bar_4O4):
    n = np.size(qi)
    jacobianMatrixCq = np.zeros((n,n), dtype = float)
    identity3x3 = np.identity(3, dtype = float)
    identity2x2 = np.identity(2, dtype = float)

    # row 1-3 --> Ground constraint
    jacobianMatrixCq[0:3,0:3] = identity3x3

    # row 4-5 (r1O2 = r2O2)
    Cq45 = np.matmul(ATransformMatrixTHETA(float(qi[5])), u_bar_2O2)
    jacobianMatrixCq[3:5,3:5] = identity2x2
    jacobianMatrixCq[3:5,5:6] = Cq45

    # row 6-7 (r2A = r3A)
    Cq67_link2 = np.matmul(ATransformMatrixTHETA(float(qi[5])), u_bar_2A)
    Cq67_link3 = np.matmul(ATransformMatrixTHETA(float(qi[8])), u_bar_3A)
    jacobianMatrixCq[5:7,3:5] = identity2x2
    jacobianMatrixCq[5:7,5:6] = Cq67_link2
    jacobianMatrixCq[5:7,6:8] = -identity2x2
    jacobianMatrixCq[5:7,8:9] = -Cq67_link3

    # row 8-9 (r3B = r4B)
    Cq89_link3 = np.matmul(ATransformMatrixTHETA(float(qi[8])), u_bar_3B)
    Cq89_link4 = np.matmul(ATransformMatrixTHETA(float(qi[11])), u_bar_4B)
    jacobianMatrixCq[7:9,6:8] = identity2x2
    jacobianMatrixCq[7:9,8:9] = Cq89_link3
    jacobianMatrixCq[7:9,9:11] = -identity2x2
    jacobianMatrixCq[7:9,11:12] = -Cq89_link4

    # row 10-11 (r1O4 = r4O4)
    Cq1011_link1 = np.matmul(ATransformMatrixTHETA(float(qi[2])), u_bar_1O4)
    Cq1011_link4 = np.matmul(ATransformMatrixTHETA(float(qi[11])), u_bar_4O4)
    jacobianMatrixCq[9:11,0:2] = identity2x2
    jacobianMatrixCq[9:11,2:3] = Cq1011_link1
    jacobianMatrixCq[9:11,9:11] = -identity2x2
    jacobianMatrixCq[9:11,11:12] = -Cq1011_link4

    # row 12
    jacobianMatrixCq[11][5] = 1

    return jacobianMatrixCq

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
    Qd45 = np.square(float(qiDot[5]))*np.dot(ATransformMatrix(float(qi[5])), u_bar_2O)
    
    # Qd line 6-7
    Qd67a = np.square(float(qiDot[5]))*np.dot(ATransformMatrix(float(qi[5])), u_bar_2A)
    Qd67b = np.square(float(qiDot[8]))*np.dot(ATransformMatrix(float(qi[8])), u_bar_3A)
    Qd67 = Qd67a-Qd67b

    # Qd line 8-9
    Qd89a = np.square(float(qi[8]))*np.dot(ATransformMatrix(float(qi[8])), u_bar_3B)
    Qd89b = np.square(float(qi[11]))*np.dot(ATransformMatrix(float(qi[11])), u_bar_4B)
    Qd89 = Qd89a-Qd89b

    # Qd line 10-11
    Qd1011a = np.square(float(qi[2]))*np.dot(ATransformMatrix(float(qi[2])), u_bar_1O4)
    Qd1011b = np.square(float(qi[11]))*np.dot(ATransformMatrix(float(qi[11])), u_bar_4O4)
    Qd1011 = Qd1011a-Qd1011b


    Qd = np.array([[0],[0],[0], Qd45[0], Qd45[1],
                    Qd67[0], Qd67[1], Qd89[0], Qd89[1], 
                    Qd1011[0], Qd1011[1], [0]])
    
    inverse_jacobian = np.linalg.inv(jacobianMatrix)
    qiDotDot= np.dot(inverse_jacobian, Qd)

    return qiDotDot

def eulerMethod(initialState, time, stepSize, systemFunction):
    # initialState = y1 and y2 (e.g. position and velocity)
    totalData = np.size(time)
    position = np.zeros(totalData, dtype = float)
    velocity = np.zeros(totalData, dtype = float)
    acceleration = np.zeros(totalData, dtype = float)

    state = initialState

    for i in range(totalData):
        eulerMeth = systemFunction(time[i], state)
        state = state + stepSize*np.array([state[1], [eulerMeth]], dtype = float)

        position[i] = (state[0])
        velocity[i] = (state[1])
        acceleration[i] = eulerMeth
    
    return position, velocity, acceleration

def rungeKutta4 (y, time, systemFunction, stepSize):
    # This function works with ANY number of DOF

    numberOfDOF = int(np.size(y)/2)
    totalData   = np.size(time)
    state       = np.zeros((totalData,numberOfDOF), dtype = float)
    stateDot    = np.zeros((totalData,numberOfDOF), dtype = float)
    stateDotDot = np.zeros((totalData,numberOfDOF), dtype = float)
    
    # RungeKutta4
    for currentTime in range(totalData):
        
        t1  = time[currentTime]
        f_1 = systemFunction(y, t1)
        k1  = np.zeros((np.size(y), 1), dtype = float)        
        for x    in range(numberOfDOF):
            k1[x]   = y[x+numberOfDOF]
            k1[x+numberOfDOF] = f_1[x]

        t2  = time[currentTime]+0.5*stepSize
        y2  = y + 0.5*k1*stepSize
        f_2 = systemFunction(y2, t2)
        k2  = np.zeros((np.size(y), 1), dtype = float)           
        for x    in range(numberOfDOF):
            k2[x]   = y2[x+numberOfDOF]
            k2[x+numberOfDOF] = f_2[x]
        
        t3  = time[currentTime]+0.5*stepSize
        y3  = y + 0.5*k2*stepSize
        f_3 = systemFunction(y3, t3)
        k3  = np.zeros((np.size(y), 1), dtype = float)           
        for x    in range(numberOfDOF):
            k3[x]   = y3[x+numberOfDOF]
            k3[x+numberOfDOF] = f_3[x]

        t4  = time[currentTime]+stepSize
        y4  = y + k3*stepSize
        f_4 = systemFunction(y4, t4)
        k4  = np.zeros((np.size(y), 1), dtype = float)
        for x    in range(numberOfDOF):
            k4[x]   = y4[x+numberOfDOF]
            k4[x+numberOfDOF] = f_4[x]

        RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6

        y = y + stepSize*RKFunct
        
        for dof in range(numberOfDOF):
            state       [currentTime][dof] = y[dof]
            stateDot    [currentTime][dof] = y[dof+numberOfDOF]
            stateDotDot [currentTime][dof] = RKFunct[dof+numberOfDOF]
       
    return state, stateDot, stateDotDot
